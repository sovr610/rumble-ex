---
name: Global Workspace Competition + Broadcast + Working Memory with Ignition Dynamics
description: >-
  This skill should be used when the user asks to "implement global workspace", "add workspace competition",
  "implement ignition dynamics", "add broadcast adapters", "implement working memory", "add CfC/LTC memory",
  "implement attention competition", "add token staging", "implement slot construction",
  "add deterministic tie-breaking", "implement iterative rounds", "add convergence detection",
  "implement ignition gate", "add lock-in prevention", "implement broadcast to temporal",
  "add broadcast to symbolic", "implement broadcast to decision", "add workspace slots",
  "implement capacity-limited selection", "add novelty scoring", "implement winner decay",
  "add slot dropout", "implement ignition cooldown", "add feedback collection",
  "implement workspace state persistence", "add truncated BPTT support",
  "implement GRU fallback for workspace", "add workspace telemetry",
  or mentions global neuronal workspace theory, multi-modal token competition,
  ignition threshold dynamics, workspace broadcast packets, or CfC/LTC working memory
  in the cognitive pipeline.
version: 0.1.0
---

# Global Workspace Competition + Broadcast + Working Memory with Ignition Dynamics

## Purpose

This skill standardizes the attention + limited-capacity + global broadcast control layer:
taking multi-modal token streams, running capacity-limited competition with deterministic
tie-breaking, stabilizing winners via iterative rounds, triggering "ignition" when a coalition
is stable/confident, and broadcasting selected content to specialist modules—while maintaining
persistent working-memory state across timesteps via CfC/LTC (or GRU fallback).

## Key Files

| Target Module | Template Asset | Purpose |
|---|---|---|
| `brain_ai/workspace/competition.py` | `assets/competition_template.py` | Token staging, 4-term scoring, deterministic top-K, slot construction |
| `brain_ai/workspace/global_workspace.py` | `assets/global_workspace_template.py` | Main orchestrator: iterative rounds, convergence, full pipeline |
| `brain_ai/workspace/ignition.py` | `assets/ignition_template.py` | Ignition score, gate, lock-in prevention |
| `brain_ai/workspace/broadcast.py` | `assets/broadcast_adapters_template.py` | Broadcast adapters (temporal, symbolic, decision), feedback |
| `brain_ai/workspace/working_memory.py` | `assets/working_memory_template.py` | CfC/LTC/GRU backends, state persistence, dt handling |
| `brain_ai/config.py` (extend) | `assets/workspace_config_template.py` | CompetitionConfig, RoundsConfig, IgnitionConfig, BroadcastConfig, WMConfig |

## Public Contract

All workspace interactions use a single interface:

```python
forward(encoder_outputs: Dict[str, EncoderOutput],
        state: Optional[WorkspaceState] = None,
        return_details: bool = False) -> WorkspaceOutput
reset_state(batch_size: int, device: torch.device, dtype=torch.float32) -> WorkspaceState
detach_state(state: WorkspaceState) -> WorkspaceState
```

Input `encoder_outputs` maps modality names to `EncoderOutput(feats, mask, salience, time)`.
`D == workspace_dim` always.

## WorkspaceOutput Contract

Every forward call returns:

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `slots` | `(B, K, D)` | float | Winner token embeddings (workspace slots) |
| `slot_mask` | `(B, K)` | bool | Valid slots per batch item |
| `winners` | WinnersMetadata | — | modality_ids, local_ids, scores per slot |
| `ignition_score` | `(B,)` | float | Ignition scalar in [0, 1] |
| `ignited` | `(B,)` | bool | Whether ignition fired |
| `broadcast_packets` | Dict[str, BroadcastPacket] | — | Per-specialist adapted tensors |
| `workspace_state` | WorkspaceState | — | Persistent WM state + ignition state |
| `telemetry` | Optional[Dict] | — | Round history, attention, convergence metrics |

## Competition Pipeline

Capacity-limited selection with fixed modality ordering and deterministic tie-breaking.

| Step | Operation | Output |
|---|---|---|
| Stage | Concatenate modality tokens in fixed order | `(B, T_total, D)` token table |
| Score | `w_content*f + w_salience*s + w_novelty*n + w_task*t` | `(B, T_total)` scores |
| Gate | Top-K with epsilon tie-breaking | `(B, K)` winner indices |
| Construct | Gather winners + optional slot mixer | `(B, K, D)` slots |

Key rules:
- Fixed modality order: `["vision", "text", "audio", "sensors", "engram"]`
- Deterministic tie-breaking: `score_adj = score + eps1*(-mod_id) + eps2*(-local_id)`
- All scoring in fp32 (tie-breaking epsilon requires precision)
- Masked tokens receive `-inf` score before top-k

## Iterative Rounds

Selection → broadcast → re-competition until stable:

| Metric | Formula | Threshold |
|---|---|---|
| Winner set stability | `Jaccard(winners_r, winners_{r-1})` | `>= 0.9` |
| Slot embedding stability | `mean_cosine(slots_r, slots_{r-1})` | `>= 0.95` |

Early-stop when both metrics exceed thresholds for M consecutive rounds.
Cap at `max_rounds` (default 4, range 2–6).

## Ignition Dynamics

Non-linear gated broadcast triggered by stable, confident competition.

### Ignition Score

```python
ignition = w_stability * stability + w_confidence * confidence
         + w_margin * margin + w_coherence * coherence
```

### Ignition Gate

| Condition | Broadcast Gain | Memory Write |
|---|---|---|
| `ignition >= threshold` (committed) | 1.0 (full) | Full persistence |
| `ignition < threshold` (weak) | `weak_gain` (e.g., 0.3) | Partial / conservative |

Training: smooth sigmoid gate. Inference: hard threshold.

### Lock-In Prevention

- **Novelty term**: `1 - cosine(token, wm_summary)` prevents same winners
- **Slot dropout**: training-only random slot masking
- **Winner decay**: previous winners get score penalty
- **Cooldown**: threshold temporarily raised after ignition

## Broadcast Adapters

Explicit modules converting slots to specialist-specific formats:

| Adapter | Output Shape | Consumer |
|---|---|---|
| `BroadcastToTemporal` | `(B, K, D)` | HTM temporal memory |
| `BroadcastToPooled` | `(B, D)` | General downstream |
| `BroadcastToSymbolic` | `(B, N_pred, D_pred)` | Reasoning module |
| `BroadcastToDecision` | `(B, D_decision)` | Active inference |

Adapters emit aligned masks and support feedback collection from specialists.

## Working Memory

Maintains persistent state across timesteps with automatic backend selection.

| Backend | Library | Strengths | Fallback |
|---|---|---|---|
| CfC | ncps | Continuous-time, irregular dt | Primary |
| LTC | ncps | More expressive, long-range | Secondary |
| GRU | torch.nn | Always available | Fallback |

Key requirements:
- `reset_state(batch_size, device, dtype)` creates fresh state
- `detach_state(state)` for truncated BPTT
- Ignition-gated write: `next_state = gain * new + (1-gain) * old`
- dt passed as timespans to CfC/LTC; concat/gate_scale for GRU
- Downstream code never branches on backend type

## Configuration Surface

### CompetitionConfig

| Field | Default | Purpose |
|---|---|---|
| `capacity_limit` | 7 | K workspace slots (Miller's Law) |
| `num_heads` | 16 | Multi-head attention heads |
| `w_content` | 1.0 | Content scoring weight |
| `w_salience` | 0.5 | Salience scoring weight |
| `w_novelty` | 0.3 | Novelty scoring weight |
| `w_task` | 0.0 | Task bias weight |
| `eps_tie_modality` | 1e-6 | Tie-breaking epsilon (modality) |
| `eps_tie_token` | 1e-8 | Tie-breaking epsilon (token) |
| `use_slot_mixer` | True | Diversity-encouraging MLP |

### IgnitionConfig

| Field | Default | Purpose |
|---|---|---|
| `ignition_threshold` | 0.3 | Ignition firing threshold |
| `w_stability` | 0.3 | Stability weight in ignition |
| `w_confidence` | 0.3 | Confidence weight |
| `weak_gain` | 0.3 | Broadcast gain when not ignited |
| `slot_dropout` | 0.1 | Training slot dropout rate |
| `winner_decay` | 0.1 | Decay for previous winners |
| `cooldown_steps` | 0 | Post-ignition cooldown |
| `w_margin` | 0.2 | Margin weight in ignition score |
| `w_coherence` | 0.2 | Cross-modal coherence weight |

See `references/ignition-dynamics.md` Section 6 for the complete field reference
including `gate_temperature`, `use_learned_ignition`, and `cooldown_boost`.

### WMConfig

| Field | Default | Purpose |
|---|---|---|
| `mode` | `"auto"` | Backend: cfc, ltc, gru, auto |
| `hidden_dim` | 4096 | Hidden state dimension |
| `buffer_capacity` | 7 | Memory buffer entries |
| `use_dt` | True | Pass time deltas to backend |
| `dt_input_mode` | `"concat"` | How GRU handles dt |

Presets: `WorkspaceConfig.minimal()`, `WorkspaceConfig.dev()`, `WorkspaceConfig.production_1b()`,
`WorkspaceConfig.production_3b()`, `WorkspaceConfig.production_7b()`.

## Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| Non-deterministic winners | Unstable tie-breaking in top-k | Use epsilon adjustment with fixed modality order |
| Workspace "stuck" on same tokens | No lock-in prevention | Enable novelty, winner decay, slot dropout |
| Ignition fires every step | Threshold too low | Raise ignition_threshold, enable cooldown |
| Ignition never fires | Threshold too high or unstable rounds | Lower threshold, increase max_rounds |
| Working memory drift | State in fp16 | Force fp32 for all WM state |
| Broadcast shape mismatch | Missing adapter for consumer | Register appropriate BroadcastAdapter |
| Iterative rounds never converge | Thresholds too strict | Lower stability_threshold, increase max_rounds |
| GRU ignoring time deltas | dt not wired through | Set use_dt=True, configure dt_input_mode |

## Anti-Patterns

- **Nondeterministic modality ordering** — use fixed MODALITY_ORDER constant
- **Relying on GPU sort stability** — always use epsilon tie-breaking
- **Pooling to (B, D) before broadcast** — keep (B, K, D) slots for adapters
- **Running > 6 iterative rounds** — latency trap; keep max_rounds <= 6
- **Using fp16 for competition scores** — tie-breaking epsilon needs fp32
- **No ignition gate** — broadcast same gain regardless of stability
- **Branching on WM backend** outside working memory module
- **Unconditional memory write** — use ignition-gated persistence
- **Forgetting detach_state** in truncated BPTT — gradient explosion

## Additional Resources

### Reference Files

- **`references/competition-broadcast.md`** — Full spec: token staging, 4-term scoring, deterministic top-K, slot construction, iterative rounds, convergence
- **`references/ignition-dynamics.md`** — Ignition score, gate, lock-in prevention, GNW theory mapping
- **`references/working-memory.md`** — CfC/LTC/GRU backends, state persistence, dt handling, ignition-gated write
- **`references/testing-matrix.md`** — All test cases: deterministic top-k, convergence, state persistence, ncps fallback

### Asset Templates

- **`assets/competition_template.py`** — TokenStager, CompetitionScorer, DeterministicTopK, SlotConstructor, self-test
- **`assets/global_workspace_template.py`** — GlobalWorkspace orchestrator, IterativeRoundManager, WorkspaceOutput, self-test
- **`assets/ignition_template.py`** — IgnitionScorer, IgnitionGate, LockInPrevention, component computers, self-test
- **`assets/broadcast_adapters_template.py`** — BroadcastAdapter base, 4 adapters, registry, FeedbackCollector, self-test
- **`assets/working_memory_template.py`** — WorkingMemory, CfC/LTC/GRU backends, MemoryBuffer, state management, self-test
- **`assets/workspace_config_template.py`** — CompetitionConfig, RoundsConfig, IgnitionConfig, BroadcastConfig, WMConfig, presets

### Scripts

- **`scripts/validate_workspace.py`** — Runtime contract validation (competition/rounds/ignition/broadcast/WM checks)
- **`scripts/gen_workspace_tests.py`** — Generates `tests/test_workspace.py` (~80+ test cases)
- **`scripts/workspace_benchmark.py`** — Performance benchmarking (throughput, latency, memory, scaling)
