# Competition-Broadcast Reference

Detailed implementation reference for the Competition and Broadcast subsystem of
the Global Workspace module. Load this when Claude needs deep implementation
details for upgrading the workspace competition pipeline, deterministic
tie-breaking, slot construction, iterative convergence, or broadcast feedback
mechanics.

---

## 1. Overview

The Competition-Broadcast pipeline is the core selection mechanism of the
Global Workspace. It takes multi-modal token streams produced by modality
encoders, runs capacity-limited competition with deterministic tie-breaking,
constructs `(B, K, D)` slot outputs, and iterates selection-broadcast rounds
until the winner set stabilizes. The pipeline replaces the current pooled
`(B, D)` output with an explicit slot-based representation that preserves
per-winner identity for downstream reasoning, HTM, and active inference
modules.

### Scope

This reference covers:

- Token staging from `EncoderOutput` with fixed modality ordering
- Four-term competition scoring (content + salience + novelty + task_bias)
- Multi-head attention and top-K gating with deterministic tie-breaking
- Slot construction with `(B, K, D)` output form and optional slot mixer
- Iterative competition-broadcast rounds with convergence detection
- Mixed-precision safety rules for competition scoring
- Migration paths from existing `AttentionCompetition`,
  `IterativeCompetition`, `RefinedBroadcast`, and
  `SelectionBroadcastWorkspace`

### Relationship to Existing Code

The existing implementation lives in `brain_ai/workspace/global_workspace.py`
(1042 lines). Two workspace implementations exist:

| Class | Lines | Status |
|---|---|---|
| `GlobalWorkspace` | 237-418 | Original, simple competition via `AttentionCompetition` |
| `SelectionBroadcastWorkspace` | 815-1013 | 2025 improved, with `IterativeCompetition` and `RefinedBroadcast` |

Supporting classes:

| Class | Lines | Role |
|---|---|---|
| `ModalityProjection` | 41-81 | Projects raw encoder features to workspace_dim with salience |
| `AttentionCompetition` | 84-186 | Single-round self-attention with soft top-K |
| `IterativeCompetition` | 555-704 | Multi-round competition with GRU-like gating and ignition detection |
| `InformationBroadcast` | 189-234 | One-directional broadcast (workspace to specialists) |
| `RefinedBroadcast` | 707-812 | Iterative broadcast with feedback projections per modality |

The upgrade addresses seven gaps:

1. No fixed modality ordering for deterministic token concatenation
2. No explicit `EncoderOutput` intake (currently uses `Dict[str, Tensor]`)
3. No 4-term competition scoring
4. No deterministic tie-breaking via epsilon method in top-K
5. No explicit `(B, K, D)` slot output form (current pools to `(B, D)`)
6. No convergence/stability metrics for early-stop
7. No slot mixer for diversity encouragement

---

## 2. Token Staging

### 2.1 Fixed Modality Ordering

Define a canonical modality order as a class-level constant. Always
concatenate encoder outputs in this order regardless of which modalities
are present or what order they arrive in. This eliminates nondeterminism
from Python dict iteration order.

```python
CANONICAL_MODALITY_ORDER: Tuple[str, ...] = (
    "vision",
    "text",
    "audio",
    "sensors",
    "engram",
)
```

Register as a class constant (not a `register_buffer` since it is not a
tensor). Place on the class body of the upgraded workspace module:

```python
class CompetitionBroadcastWorkspace(nn.Module):
    CANONICAL_MODALITY_ORDER: Tuple[str, ...] = (
        "vision", "text", "audio", "sensors", "engram",
    )

    # Mapping from modality name to integer ID for tie-breaking
    MODALITY_ID: Dict[str, int] = {
        name: idx for idx, name in enumerate(CANONICAL_MODALITY_ORDER)
    }
```

When staging tokens, iterate `CANONICAL_MODALITY_ORDER` and skip modalities
not present in the current batch:

```python
def _stage_tokens(
    self,
    encoder_outputs: List[EncoderOutput],
) -> Tuple[Tensor, Tensor, List[TokenMeta]]:
    # Index encoder outputs by modality for O(1) lookup
    by_modality: Dict[str, EncoderOutput] = {
        eo.modality: eo for eo in encoder_outputs
    }

    feats_parts: List[Tensor] = []
    mask_parts: List[Tensor] = []
    meta: List[TokenMeta] = []

    for modality_name in self.CANONICAL_MODALITY_ORDER:
        if modality_name not in by_modality:
            continue

        eo = by_modality[modality_name]
        modality_id = self.MODALITY_ID[modality_name]
        B, T_m, D = eo.feats.shape

        feats_parts.append(eo.feats)                # (B, T_m, D)
        mask_parts.append(eo.mask)                   # (B, T_m)

        for local_t in range(T_m):
            meta.append(TokenMeta(
                modality_id=modality_id,
                modality_name=modality_name,
                local_token_id=local_t,
            ))

    token_table = torch.cat(feats_parts, dim=1)      # (B, T_total, D)
    token_mask = torch.cat(mask_parts, dim=1)         # (B, T_total)

    return token_table, token_mask, meta
```

### 2.2 EncoderOutput Integration

Accept `List[EncoderOutput]` as input, not `Dict[str, Tensor]`. The
`EncoderOutput` dataclass (defined in
`brain_ai/encoders/schema.py`) provides:

```python
@dataclass
class EncoderOutput:
    modality: str                     # "vision", "text", "audio", "sensors", "engram"
    feats: Tensor                     # (B, T_m, D) float -- always 3D
    mask: Tensor                      # (B, T_m) bool -- True=valid, False=padding
    salience: Optional[Tensor]        # (B, T_m) or (B, 1) competition weight
    pos_ids: Optional[Tensor]        # (B, T_m) int64
    time: Optional[Tensor]           # (B, T_m) float32 timestamps
    spike: Optional[Tensor]          # (B, T_m, *) spike-domain
    aux: Dict[str, Any]              # diagnostics only
```

The critical invariant: `feats.shape[-1] == workspace_dim` always. Every
encoder projects into the same `D`-dimensional space before handing off to
the workspace. Assert this during token staging:

```python
for eo in encoder_outputs:
    assert eo.feats.shape[-1] == self.workspace_dim, (
        f"Encoder '{eo.modality}' feats dim {eo.feats.shape[-1]} "
        f"!= workspace_dim {self.workspace_dim}"
    )
```

Build three artifacts from the list of `EncoderOutput`:

| Artifact | Shape | Purpose |
|---|---|---|
| `token_table` | `(B, T_total, D)` | All tokens from all modalities, concatenated in canonical order |
| `token_mask` | `(B, T_total)` | Boolean validity mask, concatenated in same order |
| `token_meta` | `List[TokenMeta]` of length `T_total` | Per-token metadata mapping global index to `(modality_id, local_token_id)` |

Where `T_total = sum(T_m for each present modality)`.

### 2.3 Salience and Time Handling

Build a salience vector `(B, T_total)` by concatenating per-modality
salience values. When `EncoderOutput.salience` is `None`, default to
uniform salience of `1.0`:

```python
def _build_salience(self, encoder_outputs):
    """Concatenate salience in canonical order. Default 1.0 when None."""
    by_modality = {eo.modality: eo for eo in encoder_outputs}
    parts = []
    for name in self.CANONICAL_MODALITY_ORDER:
        if name not in by_modality:
            continue
        eo = by_modality[name]
        B, T_m = eo.feats.shape[:2]
        if eo.salience is not None:
            sal = eo.salience.expand(B, T_m) if eo.salience.shape[-1] == 1 else eo.salience
        else:
            sal = torch.ones(B, T_m, device=eo.feats.device)
        parts.append(sal)
    return torch.cat(parts, dim=1)  # (B, T_total)
```

Time handling follows the same pattern. If `EncoderOutput.time` is `None`
for a modality, use NaN sentinels for that modality's tokens. Build via
`_build_time` returning `Optional[Tensor]` of shape `(B, T_total)` or
`None` when no encoder provides timestamps. Time is only consumed by
novelty computation, not the core scoring formula.

---

## 3. Competition Scoring

### 3.1 Four-Term Scoring Formula

Compute a scalar score per token per batch item using four additive terms:

```
score(b, t) = w_content  * f_content(token_table[b, t])
            + w_salience * salience[b, t]
            + w_novelty  * novelty(token_table[b, t], wm_summary[b])
            + w_task     * task_bias[b, t]
```

All four weights (`w_content`, `w_salience`, `w_novelty`, `w_task`) come
from `CompetitionConfig` and are fixed config values, not learned
parameters. Log them at initialization for reproducibility.

#### Content Score: `f_content`

A two-layer projection that maps each token embedding to a scalar
importance score:

```python
class ContentScorer(nn.Module):
    """Score each token based on its content alone."""

    def __init__(self, workspace_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(workspace_dim, workspace_dim // 4),
            nn.LayerNorm(workspace_dim // 4),
            nn.GELU(),
            nn.Linear(workspace_dim // 4, 1),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Args:
            tokens: (B, T_total, D)
        Returns:
            scores: (B, T_total)
        """
        return self.net(tokens).squeeze(-1)
```

#### Salience Score

Directly from the encoder-provided salience vector (Section 2.3).
Already `(B, T_total)`.

#### Novelty Score

Compute how different each token is from the current working memory
summary. This prevents the same token from winning every round by
rewarding tokens that carry new information:

```python
def _compute_novelty(self, tokens, wm_summary):
    """Novelty = 1 - cosine_sim(token, wm_summary). Returns (B, T_total) in [0,1]."""
    if wm_summary is None:
        return torch.ones(tokens.shape[:2], device=tokens.device)
    cos_sim = F.cosine_similarity(tokens, wm_summary.unsqueeze(1), dim=-1)
    return (1.0 - cos_sim).clamp(0.0, 1.0)
```

#### Task Bias

An optional external bias vector `(B, T_total)` that allows downstream
task heads or active inference modules to steer competition. Default to
zeros when not provided.

#### Full Scoring

Combine all four terms:

```python
def _compute_scores(self, token_table, token_mask, salience, wm_summary, task_bias):
    """Compute 4-term competition score. Returns (B, T_total) fp32."""
    content = self.content_scorer(token_table)
    novelty = self._compute_novelty(token_table, wm_summary)
    bias = task_bias if task_bias is not None else torch.zeros_like(salience)

    scores = (self.config.w_content * content + self.config.w_salience * salience
              + self.config.w_novelty * novelty + self.config.w_task * bias)
    return scores.masked_fill(~token_mask, float('-inf'))
```

### 3.2 Score Normalization

Normalize scores before gating to prevent scale drift across iterative
rounds. Apply per-batch normalization using only valid (unmasked) tokens:

```python
def _normalize_scores(self, scores, token_mask):
    """Z-score normalize valid tokens. Invalid tokens stay at -inf."""
    valid_scores = scores.masked_fill(~token_mask, 0.0)
    count = token_mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
    mean = valid_scores.sum(dim=-1, keepdim=True) / count
    var = ((valid_scores - mean) ** 2 * token_mask.float()).sum(-1, keepdim=True) / count
    normalized = (valid_scores - mean) / (var + 1e-8).sqrt()
    return normalized.masked_fill(~token_mask, float('-inf'))
```

---

## 4. Attention + Top-K Gating

### 4.1 Multi-Head Attention

Run multi-head self-attention over the token table. Attention weights
provide a per-token importance signal that modulates the competition
scores. Use `nn.MultiheadAttention` with `batch_first=True` and apply the
token mask as a key padding mask:

```python
class CompetitionAttention(nn.Module):
    """Multi-head self-attention for competition among tokens."""
    def __init__(self, workspace_dim, num_heads=16, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=workspace_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(workspace_dim)

    def forward(self, tokens, token_mask):
        # key_padding_mask: True = ignore, so invert
        attended, attn_weights = self.attention(
            tokens, tokens, tokens,
            key_padding_mask=~token_mask,
            need_weights=True, average_attn_weights=True,
        )
        attended = self.norm(attended + tokens)  # Residual
        # Per-token importance = total attention received
        attn_importance = attn_weights.sum(dim=1)  # (B, T_total)
        return attended, attn_importance  # (B,T,D), (B,T)
```

### 4.2 Top-K Selection with Deterministic Tie-Breaking

Select `K` winners per batch item. GPU `torch.topk` uses an unstable sort,
meaning tokens with identical scores may appear in arbitrary order across
runs. Implement explicit deterministic tie-breaking using epsilon
perturbations:

```python
def _deterministic_topk(self, scores, token_meta, K,
                         eps_modality=1e-6, eps_token=1e-8):
    """Top-K with deterministic tie-breaking.
    Lower modality_id wins ties; within modality, lower local_token_id wins.
    Returns: (winner_indices (B,K), winner_scores (B,K))
    """
    device = scores.device
    max_mod = max(m.modality_id for m in token_meta) + 1
    max_tok = max(m.local_token_id for m in token_meta) + 1
    mod_ids = torch.tensor([m.modality_id for m in token_meta], dtype=torch.float32, device=device)
    tok_ids = torch.tensor([m.local_token_id for m in token_meta], dtype=torch.float32, device=device)

    # Negative: lower ID -> higher adjusted score
    tie_break = eps_modality * (-mod_ids / max_mod) + eps_token * (-tok_ids / max_tok)
    score_adj = scores + tie_break.unsqueeze(0)  # (B, T_total)

    winner_scores, winner_indices = torch.topk(
        score_adj, k=min(K, scores.shape[1]), dim=-1, largest=True, sorted=True)
    return winner_indices, winner_scores
```

**Epsilon magnitude safety argument**: The tie-breaking adjustment has
maximum magnitude `eps_modality + eps_token = 1.0006e-6`. Competition
scores from the 4-term formula (Section 3.1) are normalized (Section 3.2)
to approximately zero mean, unit variance. Two tokens with genuinely
different importance will differ by at least `O(1e-2)` after normalization.
The tie-breaking epsilon is 4 orders of magnitude smaller and cannot
perturb the true ranking.

### 4.3 Soft vs Hard Gating

During training, use soft weights via temperature-scaled softmax over
winner scores for gradient flow. During inference, use the same softmax
(gradients are not needed). Gather winner scores first:

```python
def _gated_selection(self, scores, winner_indices, training):
    """Produce gating weights for selected winners.
    Returns: gate_weights: (B, K) summing to 1.
    """
    winner_scores = scores.gather(dim=1, index=winner_indices)  # (B, K)
    gate_weights = F.softmax(
        winner_scores / self.config.competition_temperature, dim=-1
    )
    return gate_weights
```

Alternative: Gumbel top-K for differentiable selection when gradient
variance from the standard approach is too high. Add Gumbel noise
`-log(-log(U))` to scores before softmax, then take hard indices via
`torch.topk` on the perturbed scores.

---

## 5. Slot Construction

### 5.1 Gathering Winners

Construct the `(B, K, D)` slot tensor by gathering winning tokens from the
token table:

```python
def _gather_slots(self, token_table, token_mask, winner_indices):
    """Gather winning tokens into (B, K, D) slot tensor."""
    D = token_table.shape[-1]
    idx = winner_indices.unsqueeze(-1).expand(-1, -1, D)  # (B, K, D)
    slots = token_table.gather(dim=1, index=idx)           # (B, K, D)
    slot_mask = token_mask.gather(dim=1, index=winner_indices)  # (B, K)
    return slots, slot_mask
```

This produces the canonical `(B, K, D)` output that downstream modules
consume. `K` is fixed at `config.capacity_limit` (default 7, from
Miller's Law). When `T_total < K`, pad with zeros and set `slot_mask` to
`False` for the padding positions:

```python
if T_total < K:
    # Fewer tokens than slots: select all tokens, pad remaining
    winner_indices = torch.arange(T_total, device=device).unsqueeze(0).expand(B, -1)
    slots, slot_mask = self._gather_slots(token_table, token_mask, winner_indices)

    # Pad to K slots
    pad_size = K - T_total
    slots = F.pad(slots, (0, 0, 0, pad_size), value=0.0)       # (B, K, D)
    slot_mask = F.pad(slot_mask, (0, pad_size), value=False)     # (B, K)
```

### 5.2 Slot Mixer

An optional small MLP applied after slot construction to encourage
diversity among the K winning slots. Without the mixer, two semantically
similar tokens that both win may produce redundant representations. The
mixer allows cross-slot interaction via a shared feedforward network with
residual connection:

```python
class SlotMixer(nn.Module):
    """Post-selection MLP to encourage diversity among slots.

    A 2-layer FFN with residual connection applied independently to each
    slot. Despite being applied per-slot, the LayerNorm statistics are
    computed across all slots, creating implicit cross-slot interaction.
    """

    def __init__(
        self,
        workspace_dim: int,
        expansion: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden = int(workspace_dim * expansion)
        self.norm = nn.LayerNorm(workspace_dim)
        self.ffn = nn.Sequential(
            nn.Linear(workspace_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, workspace_dim),
            nn.Dropout(dropout),
        )

    def forward(self, slots: Tensor, slot_mask: Tensor) -> Tensor:
        """
        Args:
            slots: (B, K, D) winning slot embeddings
            slot_mask: (B, K) bool validity mask

        Returns:
            mixed: (B, K, D) slots after mixing
        """
        normed = self.norm(slots)
        residual = self.ffn(normed)
        mixed = slots + residual

        # Zero out invalid slots to prevent information leakage
        mixed = mixed * slot_mask.unsqueeze(-1).float()

        return mixed
```

Enable/disable via `CompetitionConfig.use_slot_mixer`. When disabled, pass
slots through unchanged. Set `self.slot_mixer = None` and skip in forward.

### 5.3 Winners Metadata

Return structured metadata about each winning slot via dataclasses:

```python
@dataclass
class TokenMeta:
    modality_id: int          # Index into CANONICAL_MODALITY_ORDER
    modality_name: str        # e.g. "vision"
    local_token_id: int       # Position within the modality's token sequence

@dataclass
class SlotInfo:
    modality_id: int
    modality_name: str
    local_token_id: int
    global_token_id: int      # Index into the concatenated token_table
    score: float              # Competition score at selection time

@dataclass
class CompetitionResult:
    slots: Tensor             # (B, K, D)
    slot_mask: Tensor         # (B, K) bool
    winner_indices: Tensor    # (B, K) int64
    winner_scores: Tensor     # (B, K) float32
    gate_weights: Tensor      # (B, K) float32, sum to 1
    slot_info: List[List[SlotInfo]]  # [batch][slot] metadata
    scores_all: Tensor        # (B, T_total) all token scores
```

Build `slot_info` by iterating `winner_indices` per batch item, looking
up each global token index in `token_meta` to populate modality and
position fields.

---

## 6. Iterative Rounds

### 6.1 Round Loop

Run competition-broadcast cycles until the winner set converges or the
maximum round count is reached. Each round refines the competition by
incorporating broadcast feedback from the previous round:

```python
def _iterative_competition(self, token_table, token_mask, token_meta,
                            salience, wm_summary, task_bias):
    K = self.config.capacity_limit
    prev_winners, prev_slots = None, None
    consecutive_stable = 0
    current_tokens = token_table

    for round_idx in range(self.config.max_rounds):
        # Score, normalize, attend
        scores = self._normalize_scores(
            self._compute_scores(current_tokens, token_mask, salience, wm_summary, task_bias),
            token_mask,
        )
        attended, attn_imp = self.competition_attention(current_tokens, token_mask)
        scores = scores + self.config.attn_importance_weight * attn_imp

        # Select, gather, gate, mix
        winner_indices, winner_scores = self._deterministic_topk(
            scores, token_meta, K, self.config.eps_tie_modality, self.config.eps_tie_token)
        slots, slot_mask = self._gather_slots(attended, token_mask, winner_indices)
        gate_weights = self._gated_selection(scores, winner_indices, self.training)
        if self.slot_mixer is not None:
            slots = self.slot_mixer(slots, slot_mask)

        # Broadcast feedback (skip last round)
        if round_idx < self.config.max_rounds - 1:
            current_tokens = self._broadcast_feedback(
                current_tokens, token_mask, slots, slot_mask, winner_indices, gate_weights)

        # Convergence check
        metrics = self._convergence_metrics(winner_indices, prev_winners, slots, prev_slots)
        if metrics['converged']:
            consecutive_stable += 1
            if consecutive_stable >= self.config.consecutive_stable_rounds:
                break
        else:
            consecutive_stable = 0
        prev_winners, prev_slots = winner_indices, slots

    return CompetitionResult(
        slots=slots, slot_mask=slot_mask, winner_indices=winner_indices,
        winner_scores=winner_scores, gate_weights=gate_weights,
        slot_info=self._build_slot_info(winner_indices, winner_scores, token_meta),
        scores_all=scores)
```

### 6.2 Broadcast Feedback

After competition, broadcast winning slots back into the token space.
Tokens consistent with the current broadcast receive a score boost;
redundant tokens get suppressed. This implements the "ignition" dynamic
where winning coalitions strengthen and losing coalitions weaken across
rounds:

```python
def _broadcast_feedback(self, tokens, token_mask, slots, slot_mask,
                         winner_indices, gate_weights):
    """Broadcast winning slots back to update token representations.
    Tokens similar to winners are reinforced; dissimilar tokens receive
    little change. Returns: updated_tokens (B, T_total, D).
    """
    B, T_total, D = tokens.shape
    # Weighted sum of winning slots -> broadcast signal (B, D)
    broadcast = (slots * gate_weights.unsqueeze(-1)).sum(dim=1)
    broadcast_exp = broadcast.unsqueeze(1).expand(-1, T_total, -1)

    # Similarity-gated update: only reinforce, not anti-reinforce
    sim = F.cosine_similarity(tokens, broadcast_exp, dim=-1).clamp(min=0.0)
    update = self.broadcast_proj(broadcast_exp) * sim.unsqueeze(-1) * self.config.broadcast_decay
    return (tokens + update) * token_mask.unsqueeze(-1).float()
```

The `broadcast_proj` is a 2-layer MLP (`Linear -> LN -> GELU -> Linear`)
initialized in the constructor.

### 6.3 Convergence Metrics

Track two stability metrics across rounds to detect convergence:

**Winner Set Stability** measures the Jaccard similarity between the
winner indices of consecutive rounds. A value of 1.0 means the exact same
tokens won in both rounds.

**Slot Embedding Stability** measures the mean cosine similarity between
corresponding slot embeddings across rounds. Even if the same tokens win,
their representations may still change due to attention updates.

```python
def _convergence_metrics(self, winners, prev_winners, slots, prev_slots):
    """Compute convergence metrics between consecutive rounds.
    Returns dict with winner_set_stability, slot_embedding_stability, converged.
    """
    if prev_winners is None or prev_slots is None:
        return {'winner_set_stability': 0.0, 'slot_embedding_stability': 0.0, 'converged': False}

    B, K = winners.shape

    # Winner set stability: mean Jaccard across batch
    stabilities = []
    for b in range(B):
        cur, prev = set(winners[b].tolist()), set(prev_winners[b].tolist())
        stabilities.append(len(cur & prev) / max(len(cur | prev), 1))
    winner_stability = sum(stabilities) / len(stabilities)

    # Slot embedding stability: sort both by index, then cosine similarity
    _, cur_order = winners.sort(dim=-1)
    _, prev_order = prev_winners.sort(dim=-1)
    cur_sorted = slots.gather(1, cur_order.unsqueeze(-1).expand(-1, -1, slots.shape[-1]))
    prev_sorted = prev_slots.gather(1, prev_order.unsqueeze(-1).expand(-1, -1, prev_slots.shape[-1]))
    embed_stability = F.cosine_similarity(cur_sorted, prev_sorted, dim=-1).mean().item()

    converged = (
        winner_stability >= self.config.stability_threshold
        and embed_stability >= self.config.embedding_stability_threshold
    )
    return {'winner_set_stability': winner_stability,
            'slot_embedding_stability': embed_stability, 'converged': converged}
```

### 6.4 Early-Stop Configuration

Configure convergence detection via `CompetitionConfig` fields:

| Field | Default | Effect |
|---|---|---|
| `max_rounds` | 4 | Maximum competition-broadcast iterations. Hard upper bound on latency. |
| `stability_threshold` | 0.9 | Minimum winner set Jaccard for convergence. |
| `embedding_stability_threshold` | 0.95 | Minimum mean cosine similarity for convergence. |
| `consecutive_stable_rounds` | 2 | Require this many consecutive stable rounds before stopping. |

The combination of `stability_threshold=0.9` and
`consecutive_stable_rounds=2` means that to converge early, the winner set
must remain 90% identical for two consecutive rounds. In practice, most
inputs converge within 2-3 rounds. Inputs with high inter-modality
competition (e.g., conflicting vision and text signals) may take all 4
rounds.

Excessive rounds (> 6) are a latency trap. Even on fast hardware, each
round adds a full attention pass plus broadcast feedback. Cap `max_rounds`
at 6 in production configurations. Log a warning if `max_rounds > 6`:

```python
if self.config.max_rounds > 6:
    logger.warning(
        f"CompetitionConfig.max_rounds={self.config.max_rounds} is high. "
        f"Each round adds O(T_total^2) attention cost. Consider max_rounds <= 6."
    )
```

---

## 7. Data Structures

### 7.1 CompetitionConfig Fields

Complete field reference for the competition configuration dataclass.
These fields supplement the existing `WorkspaceConfig` in
`brain_ai/config.py` (lines 158-190).

```python
@dataclass
class CompetitionConfig:
    """Configuration for the competition-broadcast subsystem."""

    # --- Capacity ---
    capacity_limit: int = 7
    """Maximum number of winning slots (K). Based on Miller's Law (7 +/- 2).
    Downstream modules receive (B, K, D) tensors with this K."""

    # --- Attention ---
    num_heads: int = 16
    """Number of attention heads for inter-token competition.
    Must divide workspace_dim evenly."""

    competition_temperature: float = 0.5
    """Temperature for softmax over competition scores. Lower = sharper
    winner-take-all dynamics. Higher = more distributed selection."""

    # --- Scoring weights ---
    w_content: float = 1.0
    """Weight for content-based scoring (learned projection)."""

    w_salience: float = 0.5
    """Weight for encoder-provided salience scores."""

    w_novelty: float = 0.5
    """Weight for novelty relative to working memory. Prevents perseveration."""

    w_task: float = 0.0
    """Weight for external task bias. Set to 0.0 when no task bias provided."""

    attn_importance_weight: float = 0.2
    """Weight for attention-derived importance signal added to scores."""

    # --- Iterative rounds ---
    max_rounds: int = 4
    """Maximum competition-broadcast iterations. Hard latency bound.
    Valid range: 1-6. Each round adds O(T_total^2) cost."""

    stability_threshold: float = 0.9
    """Minimum Jaccard similarity of winner sets for convergence."""

    embedding_stability_threshold: float = 0.95
    """Minimum mean cosine similarity of slot embeddings for convergence."""

    consecutive_stable_rounds: int = 2
    """Require this many consecutive converged rounds before early stop."""

    # --- Tie-breaking ---
    eps_tie_modality: float = 1e-6
    """Epsilon perturbation for modality-based tie-breaking. Lower modality_id
    gets higher adjusted score. Must be < 1e-4 to avoid score corruption."""

    eps_tie_token: float = 1e-8
    """Epsilon perturbation for within-modality token tie-breaking.
    Lower local_token_id gets higher adjusted score."""

    # --- Slot mixer ---
    use_slot_mixer: bool = True
    """Enable post-selection slot mixer MLP for diversity encouragement."""

    slot_mixer_expansion: float = 2.0
    """Hidden dimension expansion factor for the slot mixer FFN.
    Hidden dim = int(workspace_dim * expansion)."""

    # --- Broadcast ---
    broadcast_decay: float = 0.9
    """Temporal decay factor for broadcast feedback. Controls how strongly
    broadcast influences token representations in subsequent rounds.
    1.0 = full broadcast, 0.0 = no broadcast influence."""

    # --- General ---
    dropout: float = 0.1
    """Dropout rate for attention and FFN layers."""
```

### 7.2 Register Buffer Patterns

Persistent state that must survive `model.to(device)` and
`model.state_dict()` calls is registered as a buffer. Non-tensor state
(counters, histories) uses plain Python attributes.

**Buffers (register_buffer)**:

```python
# Convergence history for logging/debugging
self.register_buffer(
    '_round_count_history',
    torch.zeros(100, dtype=torch.long),
    persistent=False,  # Not saved in state_dict
)
self.register_buffer(
    '_history_idx',
    torch.tensor(0, dtype=torch.long),
    persistent=False,
)
```

**Plain attributes (not buffers)**:

```python
# Modality ordering is a class constant, not a buffer
# Scoring weights come from config, not learned
self.w_content = config.w_content   # float, not nn.Parameter
self.w_salience = config.w_salience
self.w_novelty = config.w_novelty
self.w_task = config.w_task
```

**Explicitly NOT parameters**: The scoring weights `w_content`,
`w_salience`, `w_novelty`, `w_task` are config values, not
`nn.Parameter`. Making them learnable would allow the model to collapse
the competition by zeroing out terms. Keep them as fixed hyperparameters
tuned via config presets.

---

## 8. Mixed-Precision Safety

Mixed-precision training (AMP with bfloat16 or float16) is standard for
production scale. The competition scoring path has several numerical
sensitivity points that require fp32.

### Rules

| Operation | Required Precision | Reason |
|---|---|---|
| Competition scores (all 4 terms) | fp32 | Tie-breaking epsilon requires full mantissa bits |
| Score normalization (mean, var) | fp32 | Numerical stability of variance computation |
| Tie-breaking adjustment | fp32 | Epsilon values (1e-6, 1e-8) underflow in fp16 |
| Softmax in attention | fp32 | Standard numerical best practice |
| Cosine similarity (novelty) | fp32 | Division by norms sensitive to low precision |
| Slot gathering | Matches input dtype | No precision-sensitive operations |
| Slot mixer FFN | fp16/bf16 OK | Standard FFN, not precision-sensitive |
| Broadcast feedback projection | fp16/bf16 OK | Standard projection |

### Implementation Pattern

Wrap the scoring path in `torch.amp.autocast(device_type='cuda', enabled=False)`
and cast inputs to `.float()` before computation:

```python
with torch.amp.autocast(device_type='cuda', enabled=False):
    scores = self._compute_scores(token_table.float(), token_mask,
                                   salience.float(), wm_summary, task_bias)
```

PyTorch `MultiheadAttention` internally promotes to fp32 under autocast.

---

## 9. Batch Processing

### Variable-Length Handling

All operations are batch-vectorized. Variable-length modality token counts
are handled via the `token_mask` tensor:

- `token_table` has shape `(B, T_total, D)` where `T_total` is the
  maximum total token count across the batch.
- `token_mask` has shape `(B, T_total)` with `True` for valid tokens and
  `False` for padding.
- Competition scores for masked tokens are set to `-inf` (Section 3.1) so
  they never win.
- Slot count `K` is fixed per config, not per batch item. When a batch
  item has fewer than `K` valid tokens, some slots may be invalid (marked
  in `slot_mask`).

### Batched Jaccard for Convergence

The convergence metric (Section 6.3) uses a per-batch loop for Jaccard,
acceptable because `B` is small (8-32) and `K <= 7`. For large batch
sizes, vectorize via broadcasting: expand `(B, K, 1)` vs `(B, 1, K)`,
compute match matrix, derive intersection and union from counts.

### Token Count Bounds

Document the expected token count ranges per modality for planning
`T_total`:

| Modality | Typical T_m | Max T_m (production) | Notes |
|---|---|---|---|
| vision | 1-196 | 576 | 1 for pooled, 196 for ViT 14x14, 576 for 24x24 |
| text | 1-512 | 8192 | Depends on sequence length config |
| audio | 1-300 | 1500 | Depends on utterance length |
| sensors | 1-50 | 200 | Time-series windows |
| engram | 1-16 | 64 | Memory retrieval results |

Maximum `T_total` in production: up to ~10000 tokens when all modalities
are active with long sequences. The `O(T_total^2)` attention cost makes
this the primary latency concern. Consider chunked attention or linear
attention approximations for `T_total > 4096`.

---

## 10. Migration from Existing Code

### 10.1 From AttentionCompetition

`AttentionCompetition` (lines 84-186 in `global_workspace.py`) implements
single-round competition with a gate network and soft top-K.

**Changes required**:

| Aspect | AttentionCompetition (current) | Upgraded |
|---|---|---|
| Scoring | `gate_scores + saliences` (2-term) | 4-term: content + salience + novelty + task_bias |
| Tie-breaking | None (relies on GPU sort) | Explicit epsilon method |
| Output | `winners * attention_weights` pooled to `(B, D)` | `(B, K, D)` slots |
| Rounds | Single pass | Iterative with convergence |
| Normalization | Raw scores through softmax | Explicit z-score normalization before gating |

Map existing gate network to the content scorer:

```python
# Current (AttentionCompetition.forward, line 154):
gate_scores = self.gate(attended).squeeze(-1)
combined_scores = gate_scores + saliences.squeeze(-1)

# Upgraded: replace gate with ContentScorer, add novelty + task_bias
content_scores = self.content_scorer(attended)
scores = (
    w_content * content_scores
    + w_salience * saliences
    + w_novelty * novelty
    + w_task * task_bias
)
```

The existing `self.gate` (2-layer MLP, lines 116-120) can be reused as
the initial checkpoint for `ContentScorer` since the architecture is
similar (Linear -> ReLU -> Linear -> scalar).

### 10.2 From IterativeCompetition

`IterativeCompetition` (lines 555-704) already implements multi-round
competition with GRU-like gating and ignition detection. The upgrade adds
explicit stability metrics and convergence checking.

**Changes required**:

| Aspect | IterativeCompetition (current) | Upgraded |
|---|---|---|
| Early stop | `ignition.mean() > threshold * 1.5` (line 683) | Jaccard + cosine stability for M rounds |
| Stability metrics | None | `winner_set_stability`, `slot_embedding_stability` |
| Salience update | GRUCell on features (lines 666-668) | 4-term scoring recomputed each round |
| Output | `winners * attention_weights` (line 690) | `(B, K, D)` slots via gather |
| Round history | Stores saliences + ignition | Stores stability metrics + scores |

The existing `salience_update` GRUCell and `ignition_detector` can be
retained as auxiliary signals but should not be the sole convergence
criterion. Replace the early-stop condition:

```python
# Current (line 683):
if ignition.mean() > self.ignition_threshold * 1.5:
    break

# Upgraded:
metrics = self._convergence_metrics(winner_indices, prev_winners, slots, prev_slots)
if metrics['converged']:
    consecutive_stable += 1
    if consecutive_stable >= config.consecutive_stable_rounds:
        break
```

The ignition detector remains valuable as a diagnostic signal (log it to
TensorBoard) but should not control iteration count.

### 10.3 From RefinedBroadcast

`RefinedBroadcast` (lines 707-812) implements iterative broadcast with
per-modality feedback projections. The upgrade changes the architecture
from per-modality projections to a single broadcast projection with
adapter routing.

**Changes required**:

| Aspect | RefinedBroadcast (current) | Upgraded |
|---|---|---|
| Forward projections | Per-modality `ModuleDict` (lines 736-743) | Single shared projection |
| Feedback projections | Per-modality `ModuleDict` (lines 746-752) | Adapter pattern with routing |
| Integration loop | `broadcast_iterations` fixed loops (line 780) | Absorbed into main round loop |
| Output | `(broadcasts_dict, refined_content)` | Broadcast integrated into round loop feedback |

The existing per-modality forward projections in `RefinedBroadcast` are
still needed for the final broadcast-to-specialists step (after
competition converges). Keep them but move them outside the iterative
competition loop:

```python
# After competition converges, broadcast to specialists:
final_broadcast = {}
for modality_name in self.CANONICAL_MODALITY_ORDER:
    if modality_name in self.broadcast_projections:
        # slots: (B, K, D) -> broadcast: (B, D) via weighted sum
        broadcast_signal = (
            result.slots * result.gate_weights.unsqueeze(-1)
        ).sum(dim=1)
        final_broadcast[modality_name] = self.broadcast_projections[
            modality_name
        ](broadcast_signal)
```

### 10.4 From SelectionBroadcastWorkspace

`SelectionBroadcastWorkspace` (lines 815-1013) is the main orchestrator
class. The upgrade changes its interface and internal pipeline.

**Changes required**:

| Aspect | SelectionBroadcastWorkspace (current) | Upgraded |
|---|---|---|
| Input type | `Dict[str, Tensor]` (line 914) | `List[EncoderOutput]` |
| Projection | Internal `ModalityProjection` per modality | Removed (encoders project to workspace_dim) |
| Output shape | `workspace` is `(B, D)` pooled (line 955) | `slots` is `(B, K, D)` |
| Competition | `IterativeCompetition` class | Inline `_iterative_competition` method |
| Broadcast | Separate `RefinedBroadcast` class | Inline `_broadcast_feedback` in round loop |
| Convergence | No convergence detection | Full convergence loop with metrics |

The `ModalityProjection` modules (lines 41-81) become unnecessary because
the encoder contract already guarantees `feats.shape[-1] == workspace_dim`.
Remove them to avoid double projection:

```python
# Current (SelectionBroadcastWorkspace.__init__, lines 853-859):
self.projections = nn.ModuleDict()
for name, dim in self.modality_dims.items():
    self.projections[name] = ModalityProjection(
        input_dim=dim, workspace_dim=self.config.workspace_dim, ...
    )

# Upgraded: no projections needed, encoders output at workspace_dim
# Token staging concatenates EncoderOutput.feats directly
```

The `integration` module (lines 889-894) that combines current workspace
content with previous context remains. Update it to accept slots instead
of pooled content:

```python
# Current: integrated = self.integration(cat([workspace_content, prev_context]))
# workspace_content is (B, D) from winners.sum(dim=1)

# Upgraded: pool slots first, then integrate with context
workspace_content = (
    result.slots * result.gate_weights.unsqueeze(-1)
).sum(dim=1)  # (B, D)

if self.prev_context is not None:
    combined = torch.cat([workspace_content, self.prev_context], dim=-1)
    integrated = self.integration(combined)
else:
    integrated = workspace_content
```

The `forward` method signature changes:

```python
# Current:
def forward(
    self,
    modality_inputs: Dict[str, torch.Tensor],
    modality_states: Optional[Dict[str, torch.Tensor]] = None,
    return_details: bool = False,
) -> Dict[str, torch.Tensor]:

# Upgraded:
def forward(
    self,
    encoder_outputs: List[EncoderOutput],
    wm_summary: Optional[Tensor] = None,
    task_bias: Optional[Tensor] = None,
    return_details: bool = False,
) -> WorkspaceOutput:
```

Where `WorkspaceOutput` matches the type contract defined in
`brain-ai-dev/skills/system-orchestrator/references/type-contracts.md`:

```python
@dataclass
class WorkspaceOutput:
    slots: Tensor                # (B, K, D) winning slots
    slot_mask: Tensor            # (B, K) bool
    winners: Tensor              # (B, K) indices into input
    winner_scores: Tensor        # (B, K) competition scores
    attn: Optional[Tensor]      # (B, H, K, T_total) attention maps
    broadcast: Optional[Tensor]  # (B, D) broadcast signal
    wm_state: Optional[Any]     # Working memory updated state
    modality_contributions: Optional[Dict[str, Tensor]]
```

---

## 11. Anti-Patterns

### 11.1 Nondeterministic Modality Ordering

**Wrong**: Iterating `Dict[str, Tensor]` or `Dict[str, EncoderOutput]`
directly. Python dict iteration order is insertion-order, which may vary
across runs depending on how the orchestrator populates the dict.

```python
# WRONG: nondeterministic order
for name, features in modality_inputs.items():
    projected.append(self.projections[name](features))
```

**Right**: Always iterate in `CANONICAL_MODALITY_ORDER`:

```python
# RIGHT: deterministic order
for name in self.CANONICAL_MODALITY_ORDER:
    if name in modality_inputs:
        projected.append(self.projections[name](modality_inputs[name]))
```

### 11.2 Relying on GPU Sort Stability

**Wrong**: Assuming `torch.topk` on GPU produces the same ordering for
tied scores across runs or devices.

```python
# WRONG: GPU topk is not stable
_, top_indices = torch.topk(scores, K, dim=-1)
```

**Right**: Add epsilon perturbation before `topk` (Section 4.2):

```python
# RIGHT: deterministic tie-breaking
score_adj = scores + eps_modality * (-mod_ids / max_mod) + eps_token * (-tok_ids / max_tok)
_, top_indices = torch.topk(score_adj, K, dim=-1)
```

### 11.3 Pooling Slots to (B, D) Before Downstream Use

**Wrong**: Immediately summing or averaging slots to `(B, D)` and passing
that to HTM, reasoning, and active inference. This discards per-winner
identity that downstream modules need.

```python
# WRONG: destroys slot structure
workspace = slots.sum(dim=1)  # (B, D) -- loses per-winner identity
htm_out = self.htm(workspace)
```

**Right**: Pass `(B, K, D)` slots to downstream modules. Pool only when
a specific module requires a single vector:

```python
# RIGHT: pass slots to modules that can use them
htm_out = self.htm(slots, slot_mask)  # HTM processes per-slot

# Pool only when necessary (e.g., for a classification head)
pooled = (slots * gate_weights.unsqueeze(-1)).sum(dim=1)
logits = self.classifier(pooled)
```

### 11.4 Hardcoded Modality Names in Competition Logic

**Wrong**: Using `if name == "vision"` or similar string comparisons
inside the competition scoring path.

```python
# WRONG: hardcoded modality names break extensibility
if modality_name == "vision":
    salience_boost = 1.5
elif modality_name == "text":
    salience_boost = 1.2
```

**Right**: Use the modality_id integer from `CANONICAL_MODALITY_ORDER` for
any modality-dependent behavior, and prefer config-driven weights over
hardcoded values:

```python
# RIGHT: config-driven, no string comparisons
modality_weights = self.config.per_modality_salience_scale  # Dict[str, float]
salience_boost = modality_weights.get(modality_name, 1.0)
```

### 11.5 Running > 6 Iterative Rounds

**Wrong**: Setting `max_rounds` to a large value hoping for better
convergence. Each round adds `O(T_total^2)` attention cost plus a
broadcast feedback pass.

```python
# WRONG: latency trap
config = CompetitionConfig(max_rounds=12)
```

**Right**: Cap at 6 rounds maximum. If convergence does not occur within
4 rounds, investigate whether the scoring weights or temperature need
adjustment rather than adding more rounds:

```python
# RIGHT: bounded iterations
config = CompetitionConfig(max_rounds=4)  # 2-3 rounds typical
```

### 11.6 Forgetting to Mask Invalid Tokens in Competition Scores

**Wrong**: Computing scores without masking padding tokens, allowing them
to win slots and inject garbage into downstream processing.

```python
# WRONG: padding tokens can win competition
scores = self.content_scorer(token_table)
_, top_indices = torch.topk(scores, K, dim=-1)
```

**Right**: Set invalid token scores to `-inf` before any selection
operation:

```python
# RIGHT: mask invalid tokens
scores = self.content_scorer(token_table)
scores = scores.masked_fill(~token_mask, float('-inf'))
_, top_indices = torch.topk(scores, K, dim=-1)
```

### 11.7 Using fp16 for Score Adjustments and Tie-Breaking

**Wrong**: Allowing autocast to run the scoring path in fp16. The
tie-breaking epsilon values (1e-6, 1e-8) are below fp16 precision (which
has ~3.3 decimal digits of mantissa precision, minimum representable
positive value ~6e-8).

```python
# WRONG: tie-breaking epsilons underflow in fp16
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    score_adj = scores + 1e-8 * tie_break  # 1e-8 rounds to 0 in fp16
```

**Right**: Force fp32 for the entire scoring and tie-breaking path:

```python
# RIGHT: fp32 for tie-breaking
with torch.amp.autocast(device_type='cuda', enabled=False):
    score_adj = scores.float() + 1e-8 * tie_break.float()
```

### 11.8 Modifying Slot Content After Construction

**Wrong**: Mutating slot embeddings after the mixer has run. Downstream
modules may cache or reference slot tensors, and in-place modification
causes silent correctness bugs and breaks gradient computation.

```python
# WRONG: in-place mutation after construction
slots = self.slot_mixer(slots, slot_mask)
slots[:, 0, :] *= 2.0  # In-place mutation -- breaks autograd graph
```

**Right**: Treat slots as read-only after the mixer. Any downstream
transformation should produce a new tensor:

```python
# RIGHT: create new tensor for any transformation
slots = self.slot_mixer(slots, slot_mask)
enhanced_slots = self.downstream_transform(slots)  # New tensor
```

### 11.9 Not Logging Convergence Metrics

**Wrong**: Running iterative competition without logging how many rounds
occurred and whether convergence was achieved. When debugging iterative
round behavior, the absence of metrics makes it impossible to diagnose
whether the model is converging too quickly (undershooting) or not at all
(wasting compute).

```python
# WRONG: silent iteration
for round_idx in range(max_rounds):
    winners = self._compete(tokens)
    if self._converged(winners, prev_winners):
        break
# No record of how many rounds ran or stability values
```

**Right**: Log convergence metrics to the output dict and optionally to
a logging framework:

```python
# RIGHT: track and expose metrics
round_history = []
for round_idx in range(max_rounds):
    winners = self._compete(tokens)
    metrics = self._convergence_metrics(winners, prev_winners, slots, prev_slots)
    round_history.append({
        'round': round_idx,
        'winner_set_stability': metrics['winner_set_stability'],
        'slot_embedding_stability': metrics['slot_embedding_stability'],
    })
    logger.debug(
        f"Competition round {round_idx}: "
        f"winner_stability={metrics['winner_set_stability']:.3f}, "
        f"embed_stability={metrics['slot_embedding_stability']:.3f}"
    )
    if metrics['converged']:
        break

output['competition_rounds'] = round_idx + 1
output['convergence_history'] = round_history
```

---

## Appendix A: Method Map

Summary of all methods in `CompetitionBroadcastWorkspace` with section
cross-references:

| Method | Section | Input | Output |
|---|---|---|---|
| `_stage_tokens` | 2.1 | `List[EncoderOutput]` | `(token_table, token_mask, token_meta)` |
| `_build_salience` | 2.3 | `List[EncoderOutput]` | `(B, T_total)` |
| `_build_time` | 2.3 | `List[EncoderOutput]` | `Optional[(B, T_total)]` |
| `_compute_scores` | 3.1 | `token_table, mask, salience, wm, bias` | `(B, T_total)` |
| `_normalize_scores` | 3.2 | `scores, mask` | `(B, T_total)` |
| `_compute_novelty` | 3.1 | `tokens, wm_summary` | `(B, T_total)` |
| `_deterministic_topk` | 4.2 | `scores, meta, K` | `(winner_indices, winner_scores)` |
| `_gated_selection` | 4.3 | `scores, indices, training` | `(B, K)` |
| `_gather_slots` | 5.1 | `token_table, mask, indices` | `(slots, slot_mask)` |
| `_broadcast_feedback` | 6.2 | `tokens, mask, slots, ...` | `(B, T_total, D)` |
| `_convergence_metrics` | 6.3 | `winners, prev, slots, prev` | `dict` |
| `_iterative_competition` | 6.1 | `table, mask, meta, sal, wm, bias` | `CompetitionResult` |
| `forward` | -- | `List[EncoderOutput], wm, bias` | output dict |

---

## Appendix B: Existing Code Line References

Quick-reference table mapping upgrade concerns to specific lines in the
current codebase. Use these when locating code to modify.

| File | Lines | What | Upgrade Action |
|---|---|---|---|
| `brain_ai/workspace/global_workspace.py` | 41-81 | `ModalityProjection` | Remove (encoders output at workspace_dim) |
| `brain_ai/workspace/global_workspace.py` | 84-186 | `AttentionCompetition` | Replace with `CompetitionBroadcastWorkspace` |
| `brain_ai/workspace/global_workspace.py` | 116-120 | `self.gate` MLP | Map to `ContentScorer` |
| `brain_ai/workspace/global_workspace.py` | 162-176 | Soft top-K gating | Replace with deterministic top-K (Section 4.2) |
| `brain_ai/workspace/global_workspace.py` | 183 | `winners * attention_weights` pooling | Replace with `_gather_slots` (Section 5.1) |
| `brain_ai/workspace/global_workspace.py` | 342-360 | `GlobalWorkspace.forward` projection loop | Replace with `_stage_tokens` from `List[EncoderOutput]` |
| `brain_ai/workspace/global_workspace.py` | 369 | `workspace_content = winners.sum(dim=1)` | Replace with `(B, K, D)` slot output |
| `brain_ai/workspace/global_workspace.py` | 555-704 | `IterativeCompetition` | Absorb into `_iterative_competition` with convergence |
| `brain_ai/workspace/global_workspace.py` | 666-668 | `salience_update` GRUCell | Replace with 4-term re-scoring per round |
| `brain_ai/workspace/global_workspace.py` | 683 | `ignition.mean() > threshold * 1.5` early stop | Replace with Jaccard + cosine convergence |
| `brain_ai/workspace/global_workspace.py` | 690 | `winners = current_features * attention_weights` | Replace with slot gather |
| `brain_ai/workspace/global_workspace.py` | 707-812 | `RefinedBroadcast` | Inline as `_broadcast_feedback` in round loop |
| `brain_ai/workspace/global_workspace.py` | 815-1013 | `SelectionBroadcastWorkspace` | Replace with `CompetitionBroadcastWorkspace` |
| `brain_ai/workspace/global_workspace.py` | 938-949 | Dict iteration for projections | Replace with `_stage_tokens` canonical ordering |
| `brain_ai/workspace/global_workspace.py` | 955 | `workspace_content = winners.sum(dim=1)` | Replace with slot-based output |
| `brain_ai/system.py` | 287 | `ws_output = self.workspace(encoded, ...)` | Change to pass `List[EncoderOutput]` |
| `brain_ai/system.py` | 288 | `workspace = ws_output['workspace']` | Change to `ws_output['slots']` for `(B, K, D)` |
| `brain_ai/config.py` | 158-190 | `WorkspaceConfig` | Add `CompetitionConfig` fields or nest it |
| `brain_ai/workspace/working_memory.py` | 324-362 | `WorkingMemory.forward` | Accept `(B, K, D)` slots (pool internally if needed) |
