# Integration Hooks: Neuromodulation and Eligibility Traces Across the Cognitive Pipeline

This document specifies how neuromodulation and eligibility traces integrate with every module in the `brain_ai/` architecture. Each section identifies the integration point, the signal direction, the affected weights, the configuration surface, and the pseudocode pattern. Follow these patterns when wiring three-factor plasticity into the system.

---

## 1. Architecture Overview

The brain_ai system processes data through a seven-layer cognitive pipeline:

```
Input --> Encoders --> SNN Core --> HTM --> Global Workspace --> Reasoning --> Active Inference --> Output
                                                  ^                                    |
                                            Neuromodulation  <--  Meta-Learning  <-----+
```

Neuromodulation is a cross-cutting concern. It does not occupy a single layer. Instead, it receives signals from multiple layers (reward from Active Inference, anomaly from HTM, confidence from Reasoning, spike patterns from SNN Core) and feeds plasticity updates back to designated "eligible" weights throughout the pipeline.

The four modulators -- Dopamine (DA), Acetylcholine (ACh), Norepinephrine (NE), and Serotonin (5-HT) -- are computed by the `NeuromodulatoryGate` class in `brain_ai/meta/neuromodulation.py`. Eligibility traces are maintained by `EligibilityTraceModule` (or the existing `EligibilityTrace` / `EligibilityNetwork` classes in `brain_ai/meta/eligibility.py`). The three-factor update rule combines them: `delta_w = lr * mod_signal * eligibility_trace`.

### Timing and Ordering

The integration follows a strict ordering within each forward pass:

1. **Forward sweep** (bottom-up): Input flows through Encoders, SNN Core, HTM, Workspace, Reasoning, Active Inference. During this sweep, eligibility traces accumulate from pre/post activities and signals (anomaly, confidence, reward) are collected.
2. **Modulator computation** (centralized): After the forward sweep completes, all collected signals are passed to `NeuromodulatoryGate.forward()` to produce DA, ACh, NE, and 5-HT values.
3. **Modulator broadcast** (top-down): Modulator values are dispatched to consuming modules. Modules that only receive parameter modulation (HTM learning rate, workspace temperature, reasoning threshold) apply the modulation immediately. Modules with eligible weights receive three-factor weight updates.
4. **Weight update** (local): Each eligible module multiplies its accumulated eligibility trace by the modulator signal and learning rate to produce `delta_w`, then applies it to the target weights.

This ordering ensures that modulator values reflect the complete state of the current forward pass before any weight updates occur. Do not apply three-factor updates during the forward sweep itself -- accumulate traces during the sweep and apply updates after modulator computation.

### Signal Producer and Consumer Map

```
                     PRODUCES signals for               CONSUMES modulator output
  +-----------------+----------------------------+  +-------------------------------+
  | SNN Core        | spike trains (pre, post)   |  | SNN weight matrices           |
  | HTM             | anomaly score              |  | HTM learning rate, sparsity   |
  | Workspace       | competition salience       |  | competition temperature       |
  | Reasoning       | confidence, entropy        |  | S2 iteration rate, threshold  |
  | Active Inference| reward, prediction error   |  | policy learning rate          |
  | Meta-Learning   | task embedding             |  | meta learning rate            |
  +-----------------+----------------------------+  +-------------------------------+

                     HAS eligible weights
  +-----------------------------------------+
  | SNN linear layers (SNNLinear.weight)     |
  | Fast Memory Adapters (adapter.down/up)   |
  | Reasoning MLP (S2 output_proj)           |
  | Workspace attention (optional)           |
  +-----------------------------------------+
```

---

## 2. SNN Core Integration (`core/`)

### Purpose

Apply spike-based eligibility traces to SNN synaptic weights. This is the most biologically faithful integration point -- presynaptic and postsynaptic spike trains drive an STDP-like eligibility kernel, and a dopaminergic reward signal gates the actual synaptic change.

### Integration Point

After the SNN forward pass at each timestep, before advancing to the next timestep. The target is `SNNLinear.linear.weight` for each spiking layer in `SNNCore.layers`.

### Signal Flow

```
spikes_pre (t)  --|
                  |--> EligibilityTraceModule.update(pre, post, dt)
spikes_post (t) --|
                          |
                          v
                   eligibility trace e(t)
                          |
                   mod_signal (DA from reward) --|
                                                 |--> delta_w = lr * DA * e
                                                 |
                                          weight update
```

### STDP Kernel

Use pair-based STDP timing. The kernel function is:

```
f(delta_t) = A_plus  * exp(-delta_t / tau_plus)   if delta_t > 0  (pre before post -> LTP)
           = -A_minus * exp(delta_t / tau_minus)   if delta_t < 0  (post before pre -> LTD)
```

For rate-coded SNN layers (where exact spike timing is not tracked), fall back to rate-based correlation: `f(pre, post) = outer(post, pre)`.

### Configuration

Enable via `use_snn_plasticity=True` in `ThreeFactorConfig`. Relevant fields:

| Field | Default | Purpose |
|---|---|---|
| `use_snn_plasticity` | `False` | Enable three-factor updates on SNN weights |
| `snn_trace_type` | `"accumulating"` | Trace type for SNN synapses |
| `snn_trace_decay` | `0.95` | Decay rate per timestep (match `MetaConfig.trace_decay`) |
| `snn_kernel` | `"stdp_pair"` | STDP kernel or `"rate"` for rate-based |
| `snn_stdp_tau_plus` | `20.0` | Potentiation time constant |
| `snn_stdp_tau_minus` | `20.0` | Depression time constant |

### Pattern

```python
class PlasticSNNCore(SNNCore):
    """SNNCore with three-factor eligibility trace plasticity."""

    def __init__(self, *args, three_factor_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tf_config = three_factor_config or ThreeFactorConfig()

        # Create one EligibilityTraceModule per SNNLinear layer
        self.traces = nn.ModuleList()
        for layer in self.layers:
            self.traces.append(
                EligibilityTraceModule(
                    shape=layer.linear.weight.shape,
                    trace_type=self.tf_config.snn_trace_type,
                    decay=self.tf_config.snn_trace_decay,
                    kernel=self.tf_config.snn_kernel,
                )
            )

    def forward_with_plasticity(self, x, mod_signal=None):
        """Forward pass that accumulates eligibility and optionally applies updates."""
        self.reset_mem()

        if x.dim() == 2:
            x = x.unsqueeze(0).repeat(self.num_steps, 1, 1)

        spike_record = []
        for t in range(x.shape[0]):
            input_t = x[t]
            # Layer-by-layer forward, tracking pre/post spikes
            for i, layer in enumerate(self.layers):
                pre_spikes = input_t
                spk, mem = layer(input_t)
                post_spikes = spk

                # Update eligibility trace for this layer
                self.traces[i].update(
                    pre=pre_spikes.detach(),
                    post=post_spikes.detach(),
                    dt=1.0,
                )

                # Apply three-factor update if modulator signal is available
                if mod_signal is not None:
                    layer.linear.weight = self.traces[i].apply_update(
                        weights=layer.linear.weight,
                        mod_signal=mod_signal,
                        lr=self.tf_config.lr,
                        clamp=self.tf_config.weight_clamp,
                    )

                input_t = spk  # Feed spikes to next layer

            spike_record.append(spk)

        return torch.stack(spike_record), mem
```

### Key Constraints

- Run all trace computation in fp32 even under AMP.
- Reset traces at the start of each episode (`self.traces[i].reset(batch_size, device)`).
- The `mod_signal` (DA) must be scalar or `(B,)`. If reward is delayed, accumulate eligibility across timesteps and apply the update only when reward arrives.

---

## 3. Global Workspace Integration (`workspace/`)

### Purpose

Acetylcholine (ACh) modulates the competition dynamics in the global workspace. High ACh (novel/uncertain input) sharpens the competition so that only the most salient modality wins access. Low ACh (familiar input) softens competition, allowing more diffuse information broadcast. This mirrors how attention narrows under uncertainty and broadens under familiarity.

### Integration Point

Inside `AttentionCompetition.forward()` or `IterativeCompetition.forward()`, before the softmax that produces attention weights. Modify the competition temperature based on the ACh signal.

### Signal Flow

```
HTM anomaly score --|
                    |--> NeuromodulatoryGate --> ACh signal
prediction entropy -|
                                                    |
                                                    v
                              effective_temperature = base_temperature / (1 + ach_gain * ACh)
                                                    |
                                                    v
                              attention_weights = softmax(scores / effective_temperature)
```

### Modulation Formula

```
effective_temperature = base_temperature / (1.0 + ach_gain * ACh)
```

Where:
- `base_temperature` is `competition_temperature` from `GlobalWorkspaceConfig` or `SelectionBroadcastConfig` (default 1.0 or 0.5)
- `ach_gain` is a configurable scalar (default 2.0), controlling how strongly ACh influences competition
- `ACh` is in `[0, 1]`, sourced from `NeuromodulatoryGate.forward()['modulators']['acetylcholine']`

When `ACh = 0` (fully familiar): `effective_temperature = base_temperature` (no change).
When `ACh = 1` (fully novel): `effective_temperature = base_temperature / 3.0` (sharper competition).

### Pattern

```python
class ModulatedAttentionCompetition(AttentionCompetition):
    """AttentionCompetition with ACh-modulated temperature."""

    def __init__(self, *args, ach_gain=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ach_gain = ach_gain

    def forward(self, features, saliences, mask=None, ach_signal=None):
        batch_size, num_items, _ = features.shape

        # Self-attention among modalities
        attended, raw_weights = self.attention(
            features, features, features,
            attn_mask=mask,
            need_weights=True,
        )

        # Combine with salience
        gate_scores = self.gate(attended).squeeze(-1)
        combined_scores = gate_scores + saliences.squeeze(-1)

        # ACh-modulated temperature
        if ach_signal is not None:
            # ach_signal: (batch,) or (batch, 1)
            ach = ach_signal.view(batch_size, 1)
            effective_temp = self.temperature / (1.0 + self.ach_gain * ach)
        else:
            effective_temp = self.temperature

        combined_scores = combined_scores / effective_temp

        attention_weights = F.softmax(combined_scores, dim=-1)

        # Top-K gating (unchanged)
        if num_items > self.capacity_limit:
            _, top_indices = torch.topk(attention_weights, self.capacity_limit, dim=-1)
            selection_mask = torch.zeros_like(attention_weights)
            selection_mask.scatter_(1, top_indices, 1.0)
            attention_weights = attention_weights * selection_mask
            attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + 1e-8)

        winners = attended * attention_weights.unsqueeze(-1)
        winners = self.norm(winners)

        return winners, attention_weights
```

### Optional: Rate-Based Eligibility on Workspace Attention Weights

For online adaptation of workspace attention, attach eligibility traces to the `attention` layer weights inside `AttentionCompetition`. This is optional and recommended only for streaming/online scenarios:

```python
# Create trace for attention projection weights
attn_trace = EligibilityTraceModule(
    shape=self.attention.in_proj_weight.shape,
    trace_type="accumulating",
    decay=0.97,
    kernel="rate",
)
```

Enable via `use_workspace_plasticity=True` in `ThreeFactorConfig`.

---

## 4. Reasoning Integration (`reasoning/`)

### Purpose

Dopamine (DA) modulates the confidence update rate during System 2 deliberation. Norepinephrine (NE) modulates the routing threshold between System 1 and System 2.

### DA-Gated Confidence Updates

During the System 2 iterative refinement loop in `System2Module.forward()`, DA scales how much each reasoning step updates the internal confidence:

- High DA (positive reward prediction error): increase confidence in the current reasoning trajectory. The system "trusts" its deliberation and converges faster.
- Low DA (negative RPE): decrease confidence. The system runs more S2 iterations before committing.

### NE-Modulated Routing Threshold

In `DualProcessReasoner.forward()`, the confidence threshold that determines S1-vs-S2 routing is modulated by NE:

```
effective_threshold = base_threshold - ne_sensitivity * NE
```

Where:
- `base_threshold` is `confidence_threshold` from `System2Config` (default 0.7)
- `ne_sensitivity` is configurable (default 0.2)
- `NE` is in `[0, 1]`

High NE (surprise, urgency) lowers the threshold, causing more inputs to be routed to System 2 for careful deliberation. This mirrors how arousal triggers slower, more careful thinking.

### Integration Points

1. **Metacognitive router** in `DualProcessReasoner.forward()`: NE modulates the routing decision.
2. **S2 refinement loop** in `System2Module.forward()`: DA modulates the confidence update rate between reasoning steps.

### Pattern: DA in S2 Refinement

```python
class ModulatedSystem2(System2Module):
    """System 2 with DA-modulated confidence updates."""

    def reason_step_modulated(self, state, context, da_signal=None):
        """Single deliberation step with DA modulation."""
        new_state = self.reasoning_gru(context, state)

        if self.symbolic is not None:
            symbolic_out = self.symbolic(new_state, num_steps=1)
            new_state = new_state + 0.5 * symbolic_out['output']

        # Base confidence
        confidence = self.confidence_net(new_state)

        # DA modulation: high DA -> confidence moves faster toward 1.0
        #                low DA  -> confidence moves slower / toward 0.0
        if da_signal is not None:
            da = da_signal.view(-1, 1)  # (B, 1)
            # Scale confidence update: centered at 0.5 DA baseline
            da_factor = 0.5 + da  # Range [0.5, 1.5]
            confidence = confidence * da_factor

        confidence = torch.clamp(confidence, 0.0, 1.0)
        return new_state, confidence
```

### Pattern: NE in Routing

```python
class ModulatedDualProcessReasoner(DualProcessReasoner):
    """DualProcessReasoner with NE-modulated routing threshold."""

    def __init__(self, *args, ne_sensitivity=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.ne_sensitivity = ne_sensitivity

    def forward(self, x, ne_signal=None, da_signal=None, **kwargs):
        batch_size = x.shape[0]

        sys1_output, sys1_confidence = self.system1(x)

        # NE-modulated threshold
        threshold = self.config.confidence_threshold
        if ne_signal is not None:
            ne = ne_signal.view(batch_size)
            threshold = threshold - self.ne_sensitivity * ne
            threshold = torch.clamp(threshold, min=0.3, max=0.95)
        else:
            threshold = torch.full((batch_size,), threshold, device=x.device)

        # Route based on per-sample modulated threshold
        use_system2 = sys1_confidence.squeeze(-1) < threshold

        # Continue with S2 processing where needed, passing da_signal
        # to modulate confidence updates within S2 ...
        # (remainder follows standard DualProcessReasoner logic)
```

### Optional: Rate-Based Eligibility on Reasoning MLP

Attach eligibility traces to the `output_proj` weights in `System2Module` for fast online adaptation of reasoning patterns:

```python
reasoning_trace = EligibilityTraceModule(
    shape=self.system2.output_proj[0].weight.shape,
    trace_type="accumulating",
    decay=0.95,
    kernel="rate",
)
```

Enable via `use_reasoning_plasticity=True` in `ThreeFactorConfig`.

---

## 5. HTM Integration (`temporal/`)

### Purpose

NE modulates HTM temporal memory learning rate. ACh modulates spatial pooler sparsity target. HTM anomaly score feeds back into the ACh computation (it is a signal producer, not just a consumer).

### Signal Flow

```
                          +--> NE signal --> scale TM learning rate
NeuromodulatoryGate ------|
                          +--> ACh signal --> scale SP sparsity target

HTM anomaly score --------+--> feeds INTO NeuromodulatoryGate as anomaly_score input
```

### NE Modulation of Learning Rate

High NE (surprise, urgency) increases the temporal memory learning rate so the system learns new sequences faster when aroused:

```
effective_permanence_inc = base_permanence_inc * (1.0 + ne_gain * NE)
effective_permanence_dec = base_permanence_dec * (1.0 + ne_gain * NE)
```

Where `ne_gain` defaults to 1.0. When `NE = 0`: learning rate is baseline. When `NE = 1`: learning rate doubles.

For the LSTM fallback (when `htm.core` is not available), modulate the learning rate of the LSTM optimizer or scale gradients:

```python
effective_lr = base_lr * (1.0 + ne_gain * NE.mean().item())
```

### ACh Modulation of Sparsity

High ACh (novel input) tightens the sparsity target, creating more selective representations for unfamiliar inputs:

```
effective_sparsity = base_sparsity * (1.0 - ach_sparsity_reduction * ACh)
```

Where `ach_sparsity_reduction` defaults to 0.3. When `ACh = 1`: sparsity target decreases by 30% (fewer active columns, more selective).

### Integration Point

Inside `HTMLayer.forward()` or the fallback LSTM forward pass. Modify the learning parameters before the HTM update step.

### No Direct Eligibility Traces

HTM uses its own Hebbian-like learning rules (permanence increments and decrements) rather than backpropagation. Do not attach eligibility traces to HTM internals. The integration is indirect:

1. HTM anomaly score flows INTO the `NeuromodulatoryGate` as the `anomaly_score` argument.
2. The gate computes ACh (boosted by high anomaly) and NE.
3. ACh and NE flow back to HTM to modulate its own learning parameters.

### Pattern

```python
class ModulatedHTMLayer(HTMLayer):
    """HTM layer with neuromodulatory learning rate control."""

    def __init__(self, *args, ne_gain=1.0, ach_sparsity_reduction=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.ne_gain = ne_gain
        self.ach_sparsity_reduction = ach_sparsity_reduction

    def forward(self, x, ne_signal=None, ach_signal=None):
        # Modulate learning parameters before HTM step
        if ne_signal is not None:
            ne = ne_signal.mean().item()
            self.config.permanence_inc = self._base_perm_inc * (1.0 + self.ne_gain * ne)
            self.config.permanence_dec = self._base_perm_dec * (1.0 + self.ne_gain * ne)

        if ach_signal is not None:
            ach = ach_signal.mean().item()
            effective_sparsity = self._base_sparsity * (1.0 - self.ach_sparsity_reduction * ach)
            self.config.sparsity = max(effective_sparsity, 0.005)  # Floor to prevent collapse

        # Run standard HTM forward
        result = super().forward(x)

        # anomaly score is produced here and will feed back to NeuromodulatoryGate
        return result
```

---

## 6. Meta-Learning Integration (`meta/`)

### Purpose

Three-factor eligibility updates serve as a complement (or replacement) for the MAML inner loop in online and streaming scenarios. Modulator signals also scale meta-learning rates.

### Two Integration Modes

**Option A: Eligibility alongside MAML.** The standard MAML inner loop runs as usual (gradient-based adaptation on a support set). In parallel, eligibility traces track which weights are "active" during the inner loop. The outer loop meta-update is then weighted by eligibility, giving more update to recently active synapses.

```python
# In meta-training outer loop:
for task in task_batch:
    adapted_model = inner_loop.step(model, loss_fn, support_x, support_y)

    # Compute query loss
    query_loss = loss_fn(adapted_model(query_x), query_y)

    # Standard meta-gradient
    meta_grads = torch.autograd.grad(query_loss, model.parameters())

    # Weight meta-gradient by eligibility (accumulated during inner loop)
    for param, grad, trace in zip(model.parameters(), meta_grads, traces):
        modulated_grad = grad * (1.0 + eligibility_weight * trace.trace.abs().mean())
        param.data -= outer_lr * da_signal * modulated_grad
```

**Option B: Eligibility replaces MAML inner loop.** For online (non-episodic) scenarios where there is no explicit support/query split, replace the inner loop entirely with three-factor updates. The model processes a stream of inputs, eligibility traces accumulate, and when a reward/error signal arrives, eligible weights are updated directly.

```python
# Online adaptation (no explicit task episodes):
output = model(input)
traces.update(pre=input, post=output)

if reward_available:
    mod_signal = neuromod_gate(state, reward=reward)
    for name, param in model.named_parameters():
        if name in eligible_params:
            delta_w = traces[name].get_update(mod_signal['modulators']['dopamine'], lr)
            param.data += delta_w
```

### Modulator Scaling of Meta-Learning Rate

DA scales the outer learning rate for the meta-update:

```
effective_outer_lr = base_outer_lr * (0.5 + DA)
```

High DA (positive RPE) produces a larger meta-update, reinforcing successful adaptation strategies. Low DA (negative RPE) shrinks the meta-update, preventing reinforcement of failing strategies.

ACh gates which parameters receive the meta-update. Parameters in modules processing novel inputs get larger updates:

```python
for name, param in model.named_parameters():
    module_ach = get_module_ach(name, ach_signals)
    effective_lr = base_outer_lr * (0.5 + DA.mean()) * (0.3 + 0.7 * module_ach)
    param.data -= effective_lr * meta_grad
```

### Integration Point

Inside the meta-training loop in `scripts/train_full_pipeline.py` phase 7 or in a custom `MetaTrainer` class. The eligibility traces are created for the model parameters designated in `ThreeFactorConfig.target_layers`.

---

## 7. Fast Memory Adapter Pattern

### Purpose

Rather than applying three-factor updates to the large main model weights (which risks instability), insert small dedicated adapter modules at key integration points. Only these adapter weights receive online three-factor updates. The main model weights remain frozen during online adaptation, providing stability.

### Adapter Architecture

```python
class FastMemoryAdapter(nn.Module):
    """Low-rank adapter that receives three-factor plasticity updates."""

    def __init__(self, dim, bottleneck=64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, dim, bias=False)

        # Initialize near-identity: small weights so adapter starts as passthrough
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.normal_(self.up.weight, std=0.01)

        # Eligibility traces for adapter weights
        self.trace_down = EligibilityTraceModule(
            shape=self.down.weight.shape,
            trace_type="accumulating",
            decay=0.95,
            kernel="rate",
        )
        self.trace_up = EligibilityTraceModule(
            shape=self.up.weight.shape,
            trace_type="accumulating",
            decay=0.95,
            kernel="rate",
        )

    def forward(self, x):
        """Residual adapter: output = x + up(down(x))."""
        h = self.down(x)
        adapted = self.up(h)
        return x + adapted

    def update_traces(self, x):
        """Update eligibility traces from the last forward pass."""
        h = self.down(x.detach())
        self.trace_down.update(pre=x.detach(), post=h.detach())
        self.trace_up.update(pre=h.detach(), post=self.up(h).detach())

    def apply_plasticity(self, mod_signal, lr=0.001, clamp=(-1.0, 1.0)):
        """Apply three-factor update to adapter weights."""
        self.down.weight = self.trace_down.apply_update(
            self.down.weight, mod_signal, lr=lr, clamp=clamp,
        )
        self.up.weight = self.trace_up.apply_update(
            self.up.weight, mod_signal, lr=lr, clamp=clamp,
        )

    def reset(self, batch_size, device):
        """Reset traces for new episode."""
        self.trace_down.reset(batch_size, device)
        self.trace_up.reset(batch_size, device)
```

### Insertion Points

Insert `FastMemoryAdapter` instances at three locations in the pipeline:

| Location | Position in Pipeline | Adapter Dim | Purpose |
|---|---|---|---|
| Post-encoder | After modality encoders, before workspace | `encoder.output_dim` (4096) | Adapt encoder representations online |
| Pre-workspace | Before workspace competition | `workspace.workspace_dim` (4096) | Modulate what enters workspace competition |
| Post-reasoning | After S2 reasoning output | `reasoning.hidden_dim` (4096) | Adapt reasoning output based on feedback |

### Configuration

| Field | Default | Purpose |
|---|---|---|
| `use_fast_adapters` | `True` | Enable fast memory adapters |
| `adapter_bottleneck` | 64 | Bottleneck dimension (lower = fewer params, faster) |
| `adapter_locations` | `["post_encoder", "pre_workspace", "post_reasoning"]` | Where to insert adapters |
| `adapter_lr` | 0.001 | Learning rate for adapter three-factor updates |
| `freeze_main_weights` | `True` | Freeze main model during online adaptation |

### Adapter Setup in BrainAI

```python
class BrainAIWithAdapters(BrainAI):
    """BrainAI with fast memory adapters for online three-factor plasticity."""

    def __init__(self, *args, adapter_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = adapter_config or {}
        bottleneck = cfg.get('adapter_bottleneck', 64)
        ws_dim = self.config.workspace.workspace_dim
        enc_dim = self.config.encoder.output_dim

        self.adapters = nn.ModuleDict()

        if "post_encoder" in cfg.get('adapter_locations', []):
            self.adapters['post_encoder'] = FastMemoryAdapter(enc_dim, bottleneck)
        if "pre_workspace" in cfg.get('adapter_locations', []):
            self.adapters['pre_workspace'] = FastMemoryAdapter(ws_dim, bottleneck)
        if "post_reasoning" in cfg.get('adapter_locations', []):
            self.adapters['post_reasoning'] = FastMemoryAdapter(ws_dim, bottleneck)

    def forward(self, inputs, mod_signals=None, **kwargs):
        # 1. Encode
        encoded = self.encode(inputs)

        # 2. Post-encoder adapter
        if 'post_encoder' in self.adapters:
            encoded = {k: self.adapters['post_encoder'](v) for k, v in encoded.items()}
            if mod_signals:
                for v in encoded.values():
                    self.adapters['post_encoder'].update_traces(v)

        # 3. Workspace
        # ... standard workspace processing ...

        # 4. Pre-workspace adapter (applied to projected features)
        # ... insert before competition ...

        # 5. Reasoning
        # ... standard reasoning ...

        # 6. Post-reasoning adapter
        if 'post_reasoning' in self.adapters:
            workspace = self.adapters['post_reasoning'](workspace)
            if mod_signals:
                self.adapters['post_reasoning'].update_traces(workspace)

        # Apply plasticity if modulator signals are available
        if mod_signals is not None:
            da = mod_signals['modulators']['dopamine']
            for adapter in self.adapters.values():
                adapter.apply_plasticity(da, lr=0.001)

        # Continue with decision heads ...
```

---

## 8. Signal Flow Diagram

### Complete Data Flow

This diagram shows the full lifecycle of signals, modulators, eligibility traces, and weight updates across a single forward pass.

```
 STEP 1: Forward Pass (collect signals and pre/post activities)
 ==============================================================

 Input
   |
   v
 [Encoders] -- encoded features --> [Post-Encoder Adapter] (trace accumulates)
   |
   v
 [SNN Core] -- spike trains (pre, post) --> [SNN Eligibility Traces] (trace accumulates)
   |                                          |
   v                                          | spikes_pre, spikes_post stored
 [HTM] -- anomaly_score ----+
   |                        |
   v                        |
 [Workspace Competition] <--+-- ACh modulates temperature
   |                        |
   v                        |
 [Pre-Workspace Adapter]    |    (trace accumulates)
   |                        |
   v                        |
 [Workspace Broadcast]      |
   |                        |
   v                        |
 [Reasoning (S1/S2)] ------+-- confidence, entropy
   |                   ^    |
   |                   |    +-- NE modulates S1/S2 threshold
   |                   +------- DA modulates S2 confidence update
   v
 [Post-Reasoning Adapter]   (trace accumulates)
   |
   v
 [Active Inference] -- reward, prediction_error
   |
   v
 Output


 STEP 2: Compute Modulators
 ===========================

 Collect signals from Step 1:
   - reward / prediction_error  (from Active Inference or external)
   - anomaly_score              (from HTM)
   - confidence                 (from Reasoning)
   - workspace representation   (from Workspace)

   |
   v
 [NeuromodulatoryGate.forward(x, anomaly_score, confidence, prediction_error, reward)]
   |
   +--> DA  (dopamine)        -- from reward prediction error
   +--> ACh (acetylcholine)   -- from anomaly/novelty
   +--> NE  (norepinephrine)  -- from arousal/low confidence
   +--> 5-HT (serotonin)     -- from patience/exploration balance
   +--> lr_multiplier         -- combined learning rate modifier
   +--> global_gain           -- overall responsiveness


 STEP 3: Apply Three-Factor Updates
 ====================================

 For each eligible module with accumulated traces:

   delta_w = lr * mod_signal * eligibility_trace

 Targets:
   [SNN Layers]           <-- DA * snn_eligibility       --> update SNNLinear.weight
   [Post-Encoder Adapter] <-- DA * adapter_trace_down/up  --> update adapter.down/up
   [Pre-Workspace Adapter]<-- DA * adapter_trace_down/up  --> update adapter.down/up
   [Post-Reasoning Adapter]<-- DA * adapter_trace_down/up --> update adapter.down/up
   [Reasoning MLP]        <-- DA * reasoning_trace        --> update output_proj (optional)

 Modulatory side-effects (not weight updates, but parameter modulation):
   [HTM]                  <-- NE --> scale permanence_inc/dec
   [HTM]                  <-- ACh --> scale sparsity target
   [Workspace]            <-- ACh --> scale competition temperature
   [Reasoning]            <-- NE --> scale S1/S2 routing threshold
   [Reasoning]            <-- DA --> scale S2 confidence update rate
   [Meta-Learning]        <-- DA --> scale outer learning rate


 STEP 4: Reset / Carry
 =======================

 At episode boundaries:
   - Reset all eligibility traces (or carry if configured)
   - Clear modulator history buffers
   - Reset HTM temporal memory state
   - Reset workspace prev_context
```

---

## 9. Implementation Patterns

### 9.1 Full Integration in BrainAI Forward Pass

Modify `BrainAI.forward()` in `brain_ai/system.py` to collect signals, compute modulators, and dispatch three-factor updates:

```python
def forward_with_plasticity(self, inputs, reward=None, return_details=False):
    """Forward pass with full neuromodulatory integration."""

    # --- Phase 1: Encode ---
    encoded = self.encode(inputs)

    # Apply post-encoder adapters
    if hasattr(self, 'adapters') and 'post_encoder' in self.adapters:
        encoded = {k: self.adapters['post_encoder'](v) for k, v in encoded.items()}

    # --- Phase 2: Workspace ---
    if self.workspace is not None:
        ws_output = self.workspace(encoded, return_attention=True)
        workspace = ws_output['workspace']
        anomaly_score = ws_output.get('htm', {}).get('anomaly')
    else:
        workspace = self.fallback_proj(torch.cat(list(encoded.values()), dim=-1))
        anomaly_score = None

    # --- Phase 3: Reasoning ---
    confidence = None
    if self.reasoner is not None:
        reason_out = self.reasoner(workspace, return_details=return_details)
        workspace = reason_out['output']
        confidence = reason_out['confidence']

    # --- Phase 4: Compute Modulators ---
    mod_signals = None
    if self.neuromodulation is not None:
        mod_signals = self.neuromodulation(
            workspace,
            anomaly_score=anomaly_score,
            confidence=confidence,
            reward=reward,
        )

    # --- Phase 5: Apply post-reasoning adapter ---
    if hasattr(self, 'adapters') and 'post_reasoning' in self.adapters:
        workspace = self.adapters['post_reasoning'](workspace)

    # --- Phase 6: Decision ---
    output = self.decision_heads.classify(workspace)['logits']

    # --- Phase 7: Three-factor weight updates (if training online) ---
    if mod_signals is not None and self.training:
        self._apply_three_factor_updates(mod_signals)

    return output
```

### 9.2 Per-Module Hook Registration

Use PyTorch forward hooks to automatically capture pre/post activations for eligible modules without modifying each module's forward method:

```python
class PlasticityHookManager:
    """Manages forward hooks for eligibility trace computation."""

    def __init__(self, model, eligible_modules, trace_config):
        self.model = model
        self.traces = {}
        self.activities = {}
        self.hooks = []

        for name, module in model.named_modules():
            if self._is_eligible(name, eligible_modules):
                if isinstance(module, nn.Linear):
                    trace = EligibilityTraceModule(
                        shape=module.weight.shape,
                        trace_type=trace_config.trace_type,
                        decay=trace_config.tau_e,
                        kernel=trace_config.kernel,
                    )
                    self.traces[name] = trace
                    hook = module.register_forward_hook(self._make_hook(name))
                    self.hooks.append(hook)

    def _is_eligible(self, name, eligible_modules):
        """Check if a module is designated as eligible for plasticity."""
        if eligible_modules == "all_eligible":
            return True
        return any(pattern in name for pattern in eligible_modules)

    def _make_hook(self, name):
        def hook(module, input, output):
            self.activities[name] = {
                'pre': input[0].detach(),
                'post': output.detach(),
            }
        return hook

    def update_all_traces(self):
        """Update traces for all eligible modules from captured activities."""
        for name, trace in self.traces.items():
            if name in self.activities:
                acts = self.activities[name]
                pre = acts['pre'].mean(dim=0)
                post = acts['post'].mean(dim=0)
                trace.update(pre=pre, post=post)

    def apply_all_updates(self, mod_signal, lr, clamp):
        """Apply three-factor updates to all eligible modules."""
        for name, module in self.model.named_modules():
            if name in self.traces and isinstance(module, nn.Linear):
                module.weight = self.traces[name].apply_update(
                    module.weight, mod_signal, lr=lr, clamp=clamp,
                )

    def reset_all(self, batch_size, device):
        """Reset all traces."""
        for trace in self.traces.values():
            trace.reset(batch_size, device)
        self.activities.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
```

### 9.3 Signal Collection from Multiple Sources

Centralize signal collection into a dedicated collector that gathers outputs from all pipeline stages before passing them to `NeuromodulatoryGate`:

```python
class SignalCollector:
    """Collects signals from pipeline stages for neuromodulatory computation."""

    def __init__(self):
        self.signals = {}

    def record(self, source, key, value):
        """Record a signal from a pipeline stage."""
        self.signals[f"{source}.{key}"] = value

    def get_gate_inputs(self, workspace):
        """Prepare inputs for NeuromodulatoryGate.forward()."""
        return {
            'x': workspace,
            'anomaly_score': self.signals.get('htm.anomaly'),
            'confidence': self.signals.get('reasoning.confidence'),
            'prediction_error': self.signals.get('decision.prediction_error'),
            'reward': self.signals.get('environment.reward'),
        }

    def clear(self):
        self.signals.clear()
```

Usage in the forward pass:

```python
collector = SignalCollector()

# After HTM:
collector.record('htm', 'anomaly', htm_result['anomaly'])

# After Reasoning:
collector.record('reasoning', 'confidence', reason_out['confidence'])

# After Active Inference:
collector.record('decision', 'prediction_error', ai_info.get('prediction_error'))

# External reward:
collector.record('environment', 'reward', reward)

# Compute modulators:
gate_inputs = collector.get_gate_inputs(workspace)
mod_signals = neuromod_gate(**gate_inputs)
```

### 9.4 Modulator Broadcast to Eligible Modules

After computing modulator signals, broadcast them to all modules that consume them:

```python
def broadcast_modulators(mod_signals, modules_dict):
    """Dispatch modulator signals to consuming modules."""
    da = mod_signals['modulators']['dopamine']
    ach = mod_signals['modulators']['acetylcholine']
    ne = mod_signals['modulators']['norepinephrine']
    sht = mod_signals['modulators']['serotonin']

    # Workspace: ACh controls competition temperature
    if 'workspace' in modules_dict:
        modules_dict['workspace'].set_ach_signal(ach)

    # Reasoning: DA for confidence, NE for routing
    if 'reasoner' in modules_dict:
        modules_dict['reasoner'].set_da_signal(da)
        modules_dict['reasoner'].set_ne_signal(ne)

    # HTM: NE for learning rate, ACh for sparsity
    if 'htm' in modules_dict:
        modules_dict['htm'].set_ne_signal(ne)
        modules_dict['htm'].set_ach_signal(ach)

    # Meta-learning: DA for outer LR
    if 'meta' in modules_dict:
        modules_dict['meta'].set_da_signal(da)
```

### 9.5 Complete Wiring Example

Putting it all together for a single training step with full neuromodulatory integration:

```python
def train_step_with_plasticity(brain, inputs, targets, reward, optimizer, hook_manager):
    """Single training step with three-factor plasticity."""

    # 1. Standard forward pass (backprop path)
    output = brain(inputs, return_details=True)
    loss = F.cross_entropy(output.output, targets)

    # 2. Backprop (standard gradient update for main model)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 3. Collect signals for neuromodulation
    collector = SignalCollector()
    if output.modulators is not None:
        # Modulators already computed in forward pass
        mod_signals = {'modulators': output.modulators}
    else:
        # Compute modulators from collected signals
        mod_signals = brain.neuromodulation(
            output.workspace,
            confidence=output.confidence,
            reward=reward,
        )

    # 4. Update eligibility traces from hook-captured activities
    hook_manager.update_all_traces()

    # 5. Apply three-factor updates to eligible weights (adapters, SNN synapses)
    da = mod_signals['modulators']['dopamine']
    hook_manager.apply_all_updates(
        mod_signal=da,
        lr=0.001,
        clamp=(-1.0, 1.0),
    )

    return loss.item()
```

---

## 10. Configuration Summary

### ThreeFactorConfig Fields for Integration

| Field | Default | Controls |
|---|---|---|
| `use_snn_plasticity` | `False` | Three-factor updates on SNN synaptic weights |
| `use_workspace_plasticity` | `False` | Eligibility traces on workspace attention weights |
| `use_reasoning_plasticity` | `False` | Eligibility traces on reasoning MLP weights |
| `use_fast_adapters` | `True` | Fast memory adapters with three-factor updates |
| `adapter_bottleneck` | 64 | Adapter bottleneck dimension |
| `adapter_locations` | `["post_encoder", "pre_workspace", "post_reasoning"]` | Where to insert adapters |
| `adapter_lr` | 0.001 | Learning rate for adapter plasticity |
| `ach_gain` | 2.0 | How strongly ACh modulates workspace competition |
| `ne_sensitivity` | 0.2 | How strongly NE modulates S1/S2 routing threshold |
| `ne_htm_gain` | 1.0 | How strongly NE modulates HTM learning rate |
| `ach_sparsity_reduction` | 0.3 | Max fractional reduction in HTM sparsity from ACh |
| `freeze_main_weights` | `True` | Freeze main model during online adaptation |
| `eligible_modules` | `"all_eligible"` | Module name patterns that receive three-factor updates |

### Preset Configurations

```python
@classmethod
def minimal(cls):
    """Minimal config for testing: only SNN plasticity, no adapters."""
    return cls(
        use_snn_plasticity=True,
        use_workspace_plasticity=False,
        use_reasoning_plasticity=False,
        use_fast_adapters=False,
    )

@classmethod
def dev(cls):
    """Development config: SNN plasticity + adapters, moderate settings."""
    return cls(
        use_snn_plasticity=True,
        use_fast_adapters=True,
        adapter_bottleneck=32,
        adapter_locations=["post_encoder", "post_reasoning"],
    )

@classmethod
def production(cls):
    """Full production config: all integration points active."""
    return cls(
        use_snn_plasticity=True,
        use_workspace_plasticity=True,
        use_reasoning_plasticity=True,
        use_fast_adapters=True,
        adapter_bottleneck=64,
        adapter_locations=["post_encoder", "pre_workspace", "post_reasoning"],
        ach_gain=2.0,
        ne_sensitivity=0.2,
        ne_htm_gain=1.0,
        ach_sparsity_reduction=0.3,
    )
```

---

## 11. Checklist for Adding a New Integration Point

When wiring neuromodulation into a new module, follow every step below. Skipping steps leads to silent failures (signals not flowing, traces not resetting, updates not gated).

1. **Identify consumed modulators.** Determine which of DA, ACh, NE, 5-HT the module needs. Most modules consume one or two. Document the expected input range and biological rationale.

2. **Identify produced signals.** Determine which signals the module produces for the `NeuromodulatoryGate` (anomaly, confidence, reward, prediction error, entropy). Add the signal to `SignalCollector.record()` in the forward pass.

3. **Classify the integration type.** Decide whether the module has eligible weights (three-factor weight updates via `delta_w = lr * mod * e`) or only receives parameter modulation (scaling a learning rate, temperature, or threshold). Some modules have both (e.g., Reasoning has parameter modulation for the routing threshold and optional eligibility on the MLP weights).

4. **For eligible weights:** Create `EligibilityTraceModule` instances matching the shape of each target weight tensor. Choose the trace type (accumulating, replacing, dutch) and kernel (rate, stdp_pair) based on whether the module processes spike trains or continuous activations.

5. **For parameter modulation:** Write an explicit formula mapping the modulator value to the modulated parameter. Ensure the formula is bounded -- the modulated parameter must not go negative, reach infinity, or collapse to zero. Include floor/ceiling constants. Document the formula in a comment adjacent to the code.

6. **Add configuration fields.** Extend `ThreeFactorConfig` with fields that control the new integration (enable/disable flag, gain constants, which layers are targeted). Provide sensible defaults that make the integration a no-op when disabled.

7. **Register hooks or modify forward.** For eligible weights, either register PyTorch forward hooks via `PlasticityHookManager` or modify the module's `forward()` method to store `pre_activity` and `post_activity`. Prefer hooks for cleaner separation of concerns; prefer direct modification for performance-critical inner loops (such as the SNN timestep loop).

8. **Wire signal collection.** Add the module's produced signals to `SignalCollector` at the appropriate point in `BrainAI.forward()` or `BrainAI.forward_with_plasticity()`.

9. **Wire modulator broadcast.** Add the module to `broadcast_modulators()` so it receives the modulator values it consumes.

10. **Add test cases.** At minimum, verify: (a) zero modulator produces exactly zero weight update, (b) non-zero modulator with non-zero eligibility produces a non-zero update, (c) parameter modulation stays within expected bounds across the full modulator range [0, 1], (d) traces reset correctly between episodes, (e) no cross-batch leakage in the eligibility traces.

11. **Document the integration.** Add a section to this file following the format of existing sections: Purpose, Integration Point, Signal Flow, Configuration, Pattern, Key Constraints.

12. **Verify under AMP.** Run the integration with `use_amp=True` and confirm no NaN values appear in traces, modulators, or weight updates. All trace computations must run in fp32.
