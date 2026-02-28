# Systems Consolidation Reference

## Table of Contents

1. [Overview](#1-overview)
2. [Complementary Learning Systems Theory](#2-complementary-learning-systems-theory)
3. [Hippocampal-Cortical Transfer Mapping](#3-hippocampal-cortical-transfer-mapping)
4. [Knowledge Distillation for Transfer](#4-knowledge-distillation-for-transfer)
5. [Interleaved Replay Protocol](#5-interleaved-replay-protocol)
6. [Fast-to-Slow Transfer Schedule](#6-fast-to-slow-transfer-schedule)
7. [Preventing Catastrophic Forgetting](#7-preventing-catastrophic-forgetting)
8. [brain_ai Component Mapping](#8-brain_ai-component-mapping)
9. [Appendix: Troubleshooting](#appendix-a-troubleshooting)

---

## 1. Overview

### Purpose

Systems consolidation transfers knowledge from fast-learning, episodic systems to slow-learning,
semantic systems. This is the computational analogue of the biological process where hippocampal
memories are gradually integrated into neocortical representations during sleep. In brain_ai,
this means distilling knowledge from HTM, engram memory, and episodic replay buffers into the
workspace encoder weights and transformer backbone.

### Key Principle

The fast system learns in one or few shots but has limited capacity. The slow system learns
gradually but generalizes better. Systems consolidation bridges them by using the fast system
as a teacher and the slow system as a student, trained on replayed experiences.

---

## 2. Complementary Learning Systems Theory

### Original CLS Theory

McClelland, McNaughton, and O'Reilly (1995) proposed that the brain requires two complementary
learning systems:

1. **Hippocampal system**: rapid binding of arbitrary associations, pattern-separated
   representations, sparse encoding. Learns from single exposures. High learning rate.
   Rapid forgetting without consolidation.

2. **Neocortical system**: slow extraction of statistical regularities, distributed
   representations, overlapping encoding. Learns gradually from repeated exposure. Low
   learning rate. Resistant to catastrophic interference.

### Why Two Systems?

A single system faces an impossible tradeoff:
- **High learning rate**: learns new information quickly but catastrophically overwrites
  old information (stability-plasticity dilemma).
- **Low learning rate**: preserves old information but learns new information too slowly.

The CLS solution: use the fast hippocampal system as a temporary buffer that replays
experiences to the slow neocortical system during offline periods (sleep), allowing
gradual integration without catastrophic interference.

### Modern Extensions (CLS 2.0)

Kumaran, Hassabis, and McClelland (2016) updated CLS theory to incorporate:

- **Replay with interleaving**: mixing old and new memories during replay prevents
  catastrophic forgetting of older memories.
- **Schema-consistent fast learning**: the neocortex can learn schema-consistent
  information rapidly (not just slowly).
- **Generative replay**: the hippocampus can generate novel recombinations, not just
  replay verbatim memories.

---

## 3. Hippocampal-Cortical Transfer Mapping

### Biological Transfer Process

During sleep, the hippocampus replays recently encoded memories while the neocortex is in a
receptive state (slow oscillations). This coordinated replay gradually strengthens neocortical
representations:

```
Hippocampus (fast) ──replay──> Neocortex (slow)
     |                              |
  High LR                       Low LR
  Episodic                     Semantic
  Sparse                      Distributed
  Pattern-separated            Overlapping
```

### Mapping to Neural Networks

| Biological Property | Fast System (Hippocampus-like) | Slow System (Cortex-like) |
|---|---|---|
| Learning rate | High (rapid adaptation) | Low (gradual integration) |
| Representation | Sparse, pattern-separated | Dense, distributed |
| Memory type | Episodic (specific events) | Semantic (general knowledge) |
| Capacity | Limited | Large |
| Forgetting | Rapid without consolidation | Slow, resistant |

---

## 4. Knowledge Distillation for Transfer

### Distillation Framework

The fast system acts as teacher, the slow system as student. For a replay batch:

```python
def distillation_loss(fast_output, slow_output, temperature):
    """Compute knowledge distillation loss.

    Parameters
    ----------
    fast_output : Tensor
        Logits from the fast (teacher) system. Shape (B, C).
    slow_output : Tensor
        Logits from the slow (student) system. Shape (B, C).
    temperature : float
        Softmax temperature. Higher = softer distribution = more knowledge transfer.

    Returns
    -------
    Tensor
        Scalar KL divergence loss.
    """
    fast_probs = F.softmax(fast_output / temperature, dim=-1)
    slow_log_probs = F.log_softmax(slow_output / temperature, dim=-1)
    loss = F.kl_div(slow_log_probs, fast_probs, reduction='batchmean')
    return loss * (temperature ** 2)  # scale to match original gradient magnitude
```

### Temperature Selection

- **T = 1.0**: hard targets; equivalent to standard cross-entropy on teacher predictions.
- **T = 2.0**: default; softens the distribution to reveal inter-class relationships.
- **T = 4.0+**: very soft; useful when the teacher is much larger than the student.

### Combined Loss

The total transfer loss combines distillation with a reconstruction term:

```
L_transfer = alpha * L_distillation + (1 - alpha) * L_reconstruction
```

Where:
- `L_distillation` = KL divergence between fast and slow system outputs
- `L_reconstruction` = task loss (e.g., cross-entropy on ground truth labels)
- `alpha` = `fast_to_slow_ratio` (default 0.5)

---

## 5. Interleaved Replay Protocol

### Why Interleaving Matters

If consolidation only replays recent experiences, the slow system may overfit to recent
data and forget older knowledge. Interleaving mixes experiences from different time periods:

```python
def interleaved_sample(replay_buffer, batch_size, recency_ratio=0.7):
    """Sample a batch with interleaved recent and older experiences.

    Parameters
    ----------
    replay_buffer : ReplayBuffer
        Buffer containing all stored experiences.
    batch_size : int
        Total batch size.
    recency_ratio : float
        Fraction of batch drawn from recent experiences (last 20% of buffer).

    Returns
    -------
    Batch
        Mixed batch of recent and older experiences.
    """
    n_recent = int(batch_size * recency_ratio)
    n_older = batch_size - n_recent

    # Recent: from last 20% of buffer
    recent_start = int(len(replay_buffer) * 0.8)
    recent_indices = random.sample(range(recent_start, len(replay_buffer)), n_recent)

    # Older: from first 80% of buffer
    older_indices = random.sample(range(0, recent_start), n_older)

    return replay_buffer.get(recent_indices + older_indices)
```

### Interleaving Ratios

| Setting | Recent Ratio | Older Ratio | Use Case |
|---|---|---|---|
| Aggressive recent | 0.9 | 0.1 | Early training, few old memories |
| Balanced | 0.7 | 0.3 | Default, good stability-plasticity balance |
| Conservative | 0.5 | 0.5 | Late training, many old memories to preserve |
| Preservation mode | 0.3 | 0.7 | When forgetting is detected |

---

## 6. Fast-to-Slow Transfer Schedule

### Gradual Transfer

The transfer rate should increase gradually as the fast system accumulates more reliable
knowledge:

```
transfer_weight(t) = min(1.0, t / warmup_steps) * base_transfer_weight
```

### Curriculum-Based Transfer

Transfer more from the fast system when its confidence is high:

```python
def adaptive_transfer_weight(fast_output, base_weight):
    """Scale transfer weight by teacher confidence."""
    fast_probs = F.softmax(fast_output, dim=-1)
    confidence = fast_probs.max(dim=-1).values.mean()
    return base_weight * confidence.item()
```

### Transfer Frequency

Systems consolidation need not run every sleep cycle. A recommended schedule:

| Training Phase | Transfer Frequency | Rationale |
|---|---|---|
| Phase 1-3 (early) | Every 10 sleep cycles | Fast system still learning; premature transfer may be harmful |
| Phase 4-5 (mid) | Every 5 sleep cycles | Fast system has useful knowledge; moderate transfer |
| Phase 6-7 (late) | Every 1-2 sleep cycles | Fast system well-trained; frequent transfer beneficial |

---

## 7. Preventing Catastrophic Forgetting

### Elastic Weight Consolidation (EWC) Integration

During consolidation, protect important slow-system weights using an EWC-like penalty:

```
L_total = L_transfer + lambda * sum_i(F_i * (theta_i - theta_i_old)^2)
```

Where F_i is the Fisher information for parameter i, estimated from the wake phase.

### Progressive Memory Replay

Maintain a small set of "anchor" experiences from early training that are always included
in consolidation replay. This prevents drift in foundational representations.

### Slow System Learning Rate Scheduling

Use a very low learning rate for the slow system during consolidation:

```
lr_consolidation = lr_wake * 0.1
```

This ensures gradual integration without abrupt changes.

---

## 8. brain_ai Component Mapping

### Fast System Components

| brain_ai Module | CLS Role | Properties |
|---|---|---|
| HTM Spatial Pooler | Fast pattern recognition | Rapid column activation, sparse distributed representation |
| HTM Temporal Memory | Fast sequence learning | One-shot temporal pattern encoding |
| Engram Memory | Fast episodic storage | Hash-based rapid retrieval, N-gram associations |
| Replay Buffer | Hippocampal buffer | Stores recent experiences for offline replay |

### Slow System Components

| brain_ai Module | CLS Role | Properties |
|---|---|---|
| Workspace Encoder | Slow feature extraction | Gradual refinement of sensory representations |
| Global Workspace | Slow integration | Broadcast and competition across modalities |
| Transformer Backbone | Slow sequence modeling | Deep contextual representations |
| Active Inference | Slow world model | Predictive model of environment dynamics |

### Transfer Pathways

```
HTM (fast sequence) ──distill──> Transformer (slow sequence)
Engram (fast lookup) ──distill──> Workspace Encoder (slow features)
Replay Buffer ──replay──> All slow systems (interleaved training)
```

---

## Appendix A: Troubleshooting

| Issue | Cause | Resolution |
|---|---|---|
| Slow system performance degrades | Transfer LR too high; catastrophic interference | Reduce transfer_learning_rate; add EWC penalty |
| Fast system forgets after transfer | Transfer modifies fast system (it should not) | Ensure fast system is frozen during transfer |
| Distillation loss doesn't decrease | Temperature too low; fast/slow output mismatch | Increase temperature; verify output dimensions match |
| Transfer has no effect | Transfer weight too low or too few steps | Increase fast_to_slow_ratio; increase transfer steps |
| Interleaving makes training unstable | Too many old experiences; distribution shift | Reduce older ratio; ensure old experiences are still relevant |
| OOM during distillation | Both models loaded simultaneously | Use gradient checkpointing; process fast model inference separately |
