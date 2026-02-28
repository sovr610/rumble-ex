# Chinchilla Scaling Laws: Comprehensive Reference

## Overview

This reference covers the Chinchilla paper's core findings on compute-optimal training, the derivation of key metrics, known limitations, and post-publication developments. The planner uses these principles as its theoretical backbone.

---

## 1. Core Finding: Proportional Scaling

**Hoffmann et al. (2022), "Training Compute-Optimal Large Language Models"** (DeepMind)

The central finding: **under a fixed compute budget, model parameters and training tokens should be scaled proportionally**. Prior to this work, the dominant heuristic (from Kaplan et al., 2020) suggested that compute budgets were best spent on larger models with fewer tokens. Chinchilla demonstrated the opposite.

> "Doubling parameters should be accompanied by doubling tokens."

### The 70B Reference Point

Chinchilla (70B parameters, 1.4T tokens) was trained as a counter-example to Gopher (280B parameters, ~300B tokens). Despite having **4x fewer parameters**, Chinchilla matched or outperformed Gopher across most benchmarks. This is the canonical data point:

```
N = 70e9  params
D = 1.4e12 tokens
tokens_per_param = D / N = 1.4e12 / 70e9 = 20.0
```

This "20 tokens per parameter" figure became the widely-cited rule of thumb for compute-optimal dense Transformer pretraining.

---

## 2. The `tokens_per_param` Metric

`tokens_per_param` is the planner's primary diagnostic metric:

```
tokens_per_param = total_training_tokens / n_params
```

It measures how data-rich a training run is relative to model size. Chinchilla's empirical finding sets the compute-optimal target at **~20** for standard dense decoder-only Transformers.

### Why It Matters

- Below target: the model is **undertrained** — it would improve meaningfully with more data at the same parameter count.
- Above target (significantly): the run is in the **"overtrain" regime** — acceptable if inference cost dominates (see Section 6).
- At target: the run sits on the compute-optimal frontier.

---

## 3. Kaplan vs. Chinchilla: A Critical Contrast

| Aspect | Kaplan et al. (2020) | Hoffmann et al. (2022) |
|--------|----------------------|------------------------|
| Optimal strategy | Larger model, fewer tokens | Proportional scale |
| Key metric | Loss vs. compute, fixing N large | Loss minimized over (N, D) jointly |
| Data emphasis | Underemphasized | Strongly emphasized |
| Representative | GPT-3 (175B, ~300B tokens) | Chinchilla (70B, 1.4T tokens) |
| Conclusion | Scale params aggressively | Scale tokens equally |

**Why the difference?** Kaplan et al. fixed learning rate schedules and did not jointly optimize over (N, D). Hoffmann et al. ran isoFLOP comparisons where for a given compute budget C, many (N, D) pairs satisfying C = k*N*D were compared empirically.

**Implication for the planner**: The planner makes the underlying assumption (`tokens_per_param_target`) explicit and configurable. Users who believe Kaplan-style training is appropriate for their setting can lower the target.

---

## 4. The Compute-Optimal Frontier

The compute-optimal frontier is the set of (N, D) pairs that minimize expected loss for a given compute budget C.

### Derivation

Assume:
- Total training compute: `C = k * N * D`  (k = 6 for dense Transformers)
- Optimal ratio: `D = a * N`  where `a = tokens_per_param_target`

Substituting:
```
C = k * N * (a * N) = k * a * N^2
N_opt = sqrt(C / (k * a))
D_opt = a * N_opt = a * sqrt(C / (k * a)) = sqrt(C * a / k)
```

Verification that `k * N_opt * D_opt = C`:
```
k * sqrt(C / (k*a)) * a * sqrt(C / (k*a))
= k * a * (C / (k*a))
= C  ✓
```

### Numerical Example

For C = 5e23 FLOPs, k = 6, a = 20:

```
N_opt = sqrt(5e23 / (6 * 20)) = sqrt(5e23 / 120) ≈ sqrt(4.167e21) ≈ 6.45e10 ≈ 64.5B
D_opt = 20 * 64.5e9 ≈ 1.29e12 = 1.29T tokens
```

---

## 5. The isoFLOPs Perspective

An **isoFLOPs curve** is the set of all (N, D) pairs satisfying `k * N * D = C` for a fixed C. On a log-log plot of N vs D, this is a straight line with slope -1.

The compute-optimal frontier traces through the minimum-loss point of each isoFLOPs curve. The planner's `write_svg` function renders:
1. Several isoFLOPs curves (log-spaced C values)
2. The compute-optimal frontier (the locus of optimal points)
3. The planned run as a marked point

**Reading the plot**: If the planned point is to the upper-left of the frontier (large N, small D), the model is undertrained. If it's to the lower-right (small N, large D), the model is in the overtrain regime.

---

## 6. Limitations and Scope

The Chinchilla heuristic applies most reliably to:
- Dense decoder-only Transformers
- Standard autoregressive pretraining with cross-entropy loss
- English/multilingual text corpora
- Parameter counts in the 1B–200B range (the empirically studied regime)

**Less reliable for**:

| Setting | Reason to Deviate |
|---------|-------------------|
| Instruction tuning / RLHF | Distribution shift; far fewer fine-tuning tokens needed |
| Mixture-of-Experts (MoE) | Active params != total params; routing affects FLOPs per token |
| Retrieval-augmented training | External memory changes effective capacity per FLOP |
| Long-context (>8K tokens) | Attention FLOPs scale as O(seq_len^2); k=6 underestimates |
| Very small models (<100M) | Kaplan-style scaling may be more accurate |
| Domain-specific pretraining | Data scarcity may force deviation from optimal D |

---

## 7. Post-Chinchilla Developments

### LLaMA-Style "Overtrain" Strategy

Meta's LLaMA models deliberately trained in the overtrain regime (e.g., LLaMA-1 7B on 1T tokens = ~143 tokens/param, 7x above compute-optimal). Rationale:

> "If inference cost dominates, train a smaller model for longer. A LLaMA 7B with 143 tokens/param may serve better in production than a 70B model trained compute-optimally, because the 7B is dramatically cheaper to serve."

The planner's INFO warning at `ratio > 2.0` acknowledges this regime is intentional in many modern runs.

### DeepSeek Observations

DeepSeek scaling experiments suggest that the optimal `tokens_per_param` may be higher than 20 for Chinese/multilingual corpora and for certain tokenization strategies. The planner's `tokens_per_param_target` is configurable for this reason.

### The "Inference-Optimal" vs "Training-Optimal" Distinction

- **Training-optimal**: minimize loss for a given training compute budget (Chinchilla)
- **Inference-optimal**: minimize serving cost for a given quality target (LLaMA approach)

The right choice depends on the deployment scenario:
- Research / one-time runs → training-optimal
- Production serving with millions of queries → inference-optimal (overtrain)

---

## 8. When to Deviate from the Default Target

| Scenario | Recommendation |
|----------|---------------|
| Inference cost is the primary driver | Lower target_tokens_per_param; accept overtrain |
| Data is scarce (domain-specific) | Reduce D, accept undertraining; consider repeated epochs |
| Architecture significantly differs from dense Transformer | Use custom k; validate FLOPs estimate independently |
| You are reproducing a specific paper's run | Set tokens_per_param to match that paper's ratio |
| Very long sequences (>8K) | Use "transformer-aware" FLOPs mode that adds attention term |

---

## 9. Quick Reference Formulas

```python
# Core metrics
tokens_per_param = total_tokens / n_params
total_flops = k * n_params * total_tokens         # k=6 default
undertraining_ratio = tokens_per_param / tokens_per_param_target

# Solver (Mode C)
n_opt = (C / (k * a)) ** 0.5                     # a = tokens_per_param_target
d_opt = a * n_opt

# Wallclock
peak_flops_per_s = num_gpus * peak_tflops * 1e12
achieved = peak_flops_per_s * utilization
wallclock_s = total_flops / achieved
wallclock_h = wallclock_s / 3600

# Cost
cost_usd = wallclock_h * num_gpus * cost_per_gpu_hour
```

---

## 10. Canonical Data Points for Sanity-Checking

| Model | N (params) | D (tokens) | tokens/param | Source |
|-------|-----------|------------|--------------|--------|
| GPT-3 | 175B | 300B | 1.7 | Brown et al. 2020 |
| Gopher | 280B | 300B | 1.1 | Rae et al. 2021 |
| Chinchilla | 70B | 1.4T | 20.0 | Hoffmann et al. 2022 |
| LLaMA-1 7B | 7B | 1T | 142.9 | Touvron et al. 2023 |
| LLaMA-1 65B | 65B | 1.4T | 21.5 | Touvron et al. 2023 |
| LLaMA-2 70B | 70B | 2T | 28.6 | Touvron et al. 2023 |
| Mistral 7B | 7B | ~1T | ~143 | Jiang et al. 2023 |

---

## 11. Planner Configuration Implications

The planner exposes `tokens_per_param_target` (default: 20.0) so practitioners can match their chosen scaling law. Setting:
- `tokens_per_param_target=20` → Chinchilla-optimal
- `tokens_per_param_target=100+` → LLaMA-style inference-optimal
- `tokens_per_param_target=2` → Kaplan-style (large model, few tokens)

The undertraining warnings compare `planned_tokens_per_param` against whatever `tokens_per_param_target` is configured, so the planner remains useful regardless of which scaling theory the user subscribes to.
