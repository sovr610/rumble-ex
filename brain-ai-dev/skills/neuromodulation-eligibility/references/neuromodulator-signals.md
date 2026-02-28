# Neuromodulator Signals Reference

This document specifies the computation, signal mapping, bounded output guarantees, and
combination functions for the four neuromodulators used in three-factor learning:
Dopamine (DA), Acetylcholine (ACh), Norepinephrine (NE), and Serotonin (5-HT).

All four modulators are computed deterministically from observable signals. Each modulator
maps a specific class of system-state information to a bounded scalar that gates synaptic
plasticity. The modulators are then combined into a single `global_plasticity` gain that
scales the three-factor weight update `delta_w = lr * global_plasticity * e`, where `e` is
the eligibility trace.

---

## 1. Neuromodulator Overview

### Four Modulators, Four Roles

The neuromodulatory gate produces four signals, each with a distinct computational role
and biological analog:

| Modulator | Biological Analog | Computational Role | Output Range |
|-----------|-------------------|--------------------|--------------|
| DA (Dopamine) | Ventral tegmental area (VTA) reward prediction error | Reward-based synaptic strengthening/weakening | [-1, 1] |
| ACh (Acetylcholine) | Basal forebrain cholinergic nuclei | Novelty/uncertainty gating of attention and learning | [0, 1] |
| NE (Norepinephrine) | Locus coeruleus arousal system | Urgency/arousal modulation of exploration breadth | [0, 1] |
| 5-HT (Serotonin) | Dorsal raphe serotonergic nuclei | Patience/temporal discounting of credit assignment | [0, 1] |

### Design Principles

Adhere to these principles in all modulator implementations:

1. **Determinism.** Given the same inputs and the same internal state, every modulator
   produces exactly the same output. There is no stochastic sampling anywhere in the
   modulator pipeline.

2. **Bounded outputs.** DA uses `tanh` to produce values in [-1, 1]. ACh, NE, and 5-HT use
   `sigmoid` to produce values in [0, 1]. These activation functions provide natural
   saturation without explicit clamping.

3. **Semantic zero.** When no signal is available for a given modulator, that modulator
   returns its neutral value (DA = 0, ACh = 0.5, NE = 0.5, 5-HT = 0.5) rather than
   producing undefined behavior.

4. **fp32 computation.** All modulator and combination computations run in fp32 regardless of
   the AMP context. Cast inputs to fp32 at the entry point of `NeuromodulatoryGate.forward`
   and cast the output `global_plasticity` back to the caller's dtype at the exit.

5. **Batch independence.** Each item in the batch dimension receives independent modulator
   values. Running state (reward baseline, novelty history, arousal smoothing) is shared
   across the batch for efficiency but does not introduce cross-batch information leakage
   into the modulator outputs themselves.

### Signal Flow

```
                 reward ──> DA ──┐
                novelty ──> ACh ─┤
                urgency ──> NE ──┼──> combination_fn ──> global_plasticity ──> delta_w
               patience ──> 5HT ┘
```

Each modulator receives one or more raw input signals, processes them through a lightweight
computation (linear projection plus activation), and produces a scalar per batch item. The
four scalars are then fed to a configurable combination function that outputs the single
`global_plasticity` gain.

---

## 2. Dopamine (DA) -- Reward Prediction Error

### Purpose

Dopamine encodes the reward prediction error (RPE): the difference between the reward
actually received and the reward the system expected. Positive DA signals "better than
expected" and strengthens eligible synapses. Negative DA signals "worse than expected" and
weakens them. Zero DA means "as expected" and produces no synaptic change.

### Biological Analog

Dopaminergic neurons in the ventral tegmental area (VTA) and substantia nigra pars
compacta (SNc) fire phasically above baseline when outcomes exceed expectations, pause
below baseline when outcomes disappoint, and maintain tonic firing when outcomes match
predictions. This phasic response is the RPE signal used in temporal difference (TD)
learning models.

### Input Signals

DA accepts one or more of the following, depending on `NeuromodConfig.da_source`:

| Signal | Shape | Description |
|--------|-------|-------------|
| `reward` | `(B,)` or `(B, 1)` | External reward scalar |
| `td_error` | `(B,)` or `(B, 1)` | TD error from a value function (pre-computed) |
| `prediction_error` | `(B,)` or `(B, 1)` | Generic prediction error proxy |

When `da_source = "reward"` (default), use the reward signal with baseline subtraction.
When `da_source = "td_error"`, pass the TD error directly through tanh.

### Computation

**Step 1: Baseline tracking.** Maintain an exponential moving average (EMA) of recent
rewards as the expected reward baseline:

```
baseline(t) = (1 - alpha_baseline) * baseline(t-1) + alpha_baseline * reward(t)
```

where `alpha_baseline` is a hyperparameter controlling how quickly the baseline adapts.
A typical value is `alpha_baseline = 0.01` for slow tracking or `alpha_baseline = 0.1` for
fast adaptation.

**Step 2: RPE computation.** Compute the raw reward prediction error:

```
rpe = reward - baseline
```

**Step 3: Bounded output.** Apply tanh with a learnable weight to produce bounded DA:

```
DA = tanh(w_reward * rpe)
```

where `w_reward` is a learnable scalar (initialized to 1.0) that controls the sensitivity
of DA to the RPE magnitude.

### Full Formula

```
DA(t) = tanh(w_reward * (reward(t) - baseline(t)))
baseline(t) = (1 - alpha) * baseline(t-1) + alpha * reward(t)
```

### Output Range and Semantics

| DA Value | Meaning | Effect on Eligible Synapses |
|----------|---------|---------------------------|
| DA > 0 | Better than expected | Strengthen (potentiate) |
| DA = 0 | As expected | No change |
| DA < 0 | Worse than expected | Weaken (depress) |

The output is naturally bounded to [-1, 1] by the tanh activation.

### Edge Cases

- **No reward signal available.** Set DA = 0.0 (neutral). This produces zero plasticity
  from the DA channel. Do not hallucinate a reward.

- **Reward is always constant.** The baseline tracks the constant reward, so
  `rpe -> 0` and `DA -> 0`. This is correct: constant reward carries no new information.

- **Very large reward spike.** The tanh saturates at +/-1, preventing unbounded DA. The
  learnable `w_reward` can be constrained via gradient clipping or weight decay to prevent
  the tanh from operating in a near-constant regime.

- **First step (no baseline history).** Initialize `baseline = 0.0`. The first reward
  produces a strong DA signal, which is appropriate since the system has no expectations.

### Pseudocode

```python
class DopamineComputer(nn.Module):
    def __init__(self, alpha_baseline: float = 0.01):
        super().__init__()
        self.w_reward = nn.Parameter(torch.tensor(1.0))
        self.alpha_baseline = alpha_baseline
        self.register_buffer('baseline', torch.tensor(0.0))

    def forward(self, reward: torch.Tensor) -> torch.Tensor:
        """Compute DA from reward signal.

        Args:
            reward: (B,) reward values

        Returns:
            da: (B,) dopamine signal in [-1, 1]
        """
        reward = reward.float()
        # RPE = reward - expected
        rpe = reward - self.baseline
        # Bounded output
        da = torch.tanh(self.w_reward * rpe)
        return da

    def update_baseline(self, reward: torch.Tensor):
        """Update reward baseline EMA. Call once per step."""
        with torch.no_grad():
            mean_reward = reward.detach().mean()
            self.baseline.mul_(1 - self.alpha_baseline).add_(
                self.alpha_baseline * mean_reward
            )
```

---

## 3. Acetylcholine (ACh) -- Novelty/Uncertainty Gate

### Purpose

Acetylcholine encodes novelty and uncertainty. High ACh tells the system "this input is
unfamiliar or unpredictable -- pay attention and learn." Low ACh tells the system "this is
familiar territory -- rely on existing representations." ACh directly gates the effective
learning rate: when ACh is high, eligible synapses receive stronger updates.

### Biological Analog

Cholinergic projections from the basal forebrain (nucleus basalis of Meynert, medial
septum) modulate cortical attention and hippocampal encoding. High ACh release enhances
signal-to-noise in cortical circuits, promotes encoding of novel associations, and
modulates the threshold for long-term potentiation (LTP) in hippocampal synapses.

### Input Signals

ACh accepts one or more of the following uncertainty indicators:

| Signal | Shape | Description |
|--------|-------|-------------|
| `novelty` | `(B,)` or `(B, 1)` | Novelty score from a novelty detector or HTM anomaly |
| `entropy` | `(B,)` or `(B, 1)` | Entropy of the output distribution (prediction uncertainty) |
| `htm_anomaly` | `(B,)` or `(B, 1)` | HTM spatial/temporal anomaly score [0, 1] |
| `confidence` | `(B,)` or `(B, 1)` | Model confidence (inverse drives ACh up) |

When multiple signals are available, they are combined via a weighted sum before the
sigmoid. When `ach_source = "novelty"` (default), use the novelty signal alone.

### Computation

```
raw_ach = w_novelty * novelty + w_entropy * entropy + w_anomaly * htm_anomaly + bias_ach
ACh = sigmoid(raw_ach)
```

where `w_novelty`, `w_entropy`, `w_anomaly` are learnable weights (initialized to 1.0, 1.0,
1.0 respectively) and `bias_ach` is a learnable bias (initialized to 0.0).

When confidence is provided instead of novelty, convert it:

```
novelty_from_confidence = 1.0 - confidence
```

### Full Formula

```
ACh = sigmoid(sum_i(w_i * signal_i) + bias_ach)
```

where `signal_i` ranges over all available uncertainty signals.

### Output Range and Semantics

| ACh Value | Meaning | Effect on Learning |
|-----------|---------|-------------------|
| ACh -> 1.0 | Highly novel/uncertain | Maximum learning rate amplification |
| ACh ~ 0.5 | Moderate uncertainty | Baseline learning rate |
| ACh -> 0.0 | Familiar/certain | Minimal learning (consolidation mode) |

### Integration with Workspace

ACh can modulate the global workspace competition strength. When ACh is high, the
workspace should increase competition among modality specialists, allowing novel or
unexpected information to win broadcast access:

```
effective_competition_gain = base_gain * (1 + beta_ach * ACh)
```

where `beta_ach` is a scaling coefficient (default 0.5). This integration is optional and
controlled by `NeuromodConfig.ach_modulates_workspace`.

### Edge Cases

- **No novelty signal available.** Set ACh = 0.5 (neutral baseline). The sigmoid of 0 is
  0.5, so set `raw_ach = 0.0`.

- **All signals are zero.** The bias term determines ACh. If `bias_ach = 0.0`, then
  ACh = 0.5 (neutral).

- **HTM anomaly is always 1.0.** This indicates the HTM is not learning or all inputs are
  anomalous. Check that the HTM is connected and functioning. In the meantime, ACh will
  be high, which is the correct response to persistent novelty.

### Pseudocode

```python
class AcetylcholineComputer(nn.Module):
    def __init__(self, num_sources: int = 3):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_sources))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        novelty: Optional[torch.Tensor] = None,
        entropy: Optional[torch.Tensor] = None,
        htm_anomaly: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute ACh from uncertainty signals.

        Returns:
            ach: (B,) acetylcholine signal in [0, 1]
        """
        signals = []
        weights = []

        if novelty is not None:
            signals.append(novelty.float())
            weights.append(self.weights[0])
        if confidence is not None:
            signals.append(1.0 - confidence.float())
            weights.append(self.weights[0])  # Same slot as novelty
        if entropy is not None:
            signals.append(entropy.float())
            weights.append(self.weights[1])
        if htm_anomaly is not None:
            signals.append(htm_anomaly.float())
            weights.append(self.weights[2])

        if not signals:
            # No signals: return neutral
            return torch.full_like(
                next(self.parameters()), 0.5
            ).expand(1)

        raw = sum(w * s for w, s in zip(weights, signals)) + self.bias
        ach = torch.sigmoid(raw)
        return ach
```

---

## 4. Norepinephrine (NE) -- Urgency/Arousal

### Purpose

Norepinephrine encodes urgency and arousal. High NE signals "something important is
happening -- mobilize resources and broaden learning." In computational terms, high NE
increases the breadth of synaptic updates: more synapses are modified (more diffuse
plasticity) and the system shifts toward exploration.

### Biological Analog

The locus coeruleus (LC) is the primary source of norepinephrine in the brain. It responds
to salient, surprising, or threatening stimuli with phasic bursts. Tonic LC activity sets
global arousal level. The Aston-Jones model describes the LC as controlling the balance
between exploitation (low tonic NE, focused attention) and exploration (high tonic NE,
broad attention, increased behavioral variability).

### Input Signals

| Signal | Shape | Description |
|--------|-------|-------------|
| `urgency` | `(B,)` or `(B, 1)` | External urgency signal (time pressure, danger) |
| `surprise` | `(B,)` or `(B, 1)` | Absolute prediction error or KL divergence spike |
| `abs_td_error` | `(B,)` or `(B, 1)` | Absolute value of TD error (surprise proxy) |

### Computation

```
raw_ne = w_urgency * urgency + w_surprise * surprise + bias_ne
NE = sigmoid(raw_ne)
```

where `w_urgency` and `w_surprise` are learnable weights (initialized to 1.0) and `bias_ne`
is a learnable bias (initialized to 0.0).

Surprise can be derived from prediction error:

```
surprise = |prediction_error|
```

or from a KL divergence spike between consecutive belief distributions:

```
surprise = KL(q(t) || q(t-1))
```

### Temporal Smoothing

NE should be temporally smooth to prevent rapid oscillation between exploration and
exploitation modes. Apply exponential smoothing to the raw NE output:

```
NE_smooth(t) = (1 - alpha_ne) * NE_smooth(t-1) + alpha_ne * NE_raw(t)
```

where `alpha_ne` controls the smoothing rate. A typical value is `alpha_ne = 0.1` for
gradual arousal transitions or `alpha_ne = 0.5` for faster response.

### Output Range and Semantics

| NE Value | Meaning | Effect on Plasticity |
|----------|---------|---------------------|
| NE -> 1.0 | High arousal/urgency | Broad plasticity, exploration bias |
| NE ~ 0.5 | Moderate arousal | Balanced exploration/exploitation |
| NE -> 0.0 | Low arousal, calm | Focused plasticity, exploitation bias |

### Exploration vs. Exploitation

NE modulates the exploration-exploitation tradeoff in two ways:

1. **Plasticity breadth.** High NE increases the effective number of synapses receiving
   updates. Implement this by scaling the eligibility trace threshold: when NE is high,
   even weakly eligible synapses receive updates.

   ```
   effective_threshold = base_threshold * (1 - gamma_ne * NE)
   ```

   where `gamma_ne` controls how much NE broadens the update set (default 0.5).

2. **Action noise.** High NE can inject noise into the action selection process
   (e.g., increasing the temperature in softmax policy selection):

   ```
   effective_temperature = base_temperature * (1 + delta_ne * NE)
   ```

### Edge Cases

- **No urgency or surprise signal.** Set NE = 0.5 (neutral arousal). With `bias_ne = 0.0`
  and zero inputs, `sigmoid(0) = 0.5`.

- **Sustained high surprise.** The temporal smoothing prevents NE from spiking and staying
  at 1.0 indefinitely. The smoothing rate `alpha_ne` controls how quickly NE decays from
  peak arousal.

- **Extremely large surprise values.** The sigmoid saturates, bounding NE at 1.0. The
  learnable `w_surprise` weight can be regularized to keep the input to sigmoid in a
  sensitive range.

### Pseudocode

```python
class NorepinephrineComputer(nn.Module):
    def __init__(self, alpha_smooth: float = 0.1):
        super().__init__()
        self.w_urgency = nn.Parameter(torch.tensor(1.0))
        self.w_surprise = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.alpha_smooth = alpha_smooth
        self.register_buffer('ne_smooth', torch.tensor(0.5))

    def forward(
        self,
        urgency: Optional[torch.Tensor] = None,
        surprise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute NE from urgency/surprise signals.

        Returns:
            ne: (B,) norepinephrine signal in [0, 1]
        """
        raw = self.bias.clone()
        if urgency is not None:
            raw = raw + self.w_urgency * urgency.float()
        if surprise is not None:
            raw = raw + self.w_surprise * surprise.float()

        ne_raw = torch.sigmoid(raw)
        return ne_raw

    def update_smoothing(self, ne_raw: torch.Tensor):
        """Update temporal smoothing state. Call once per step."""
        with torch.no_grad():
            mean_ne = ne_raw.detach().mean()
            self.ne_smooth.mul_(1 - self.alpha_smooth).add_(
                self.alpha_smooth * mean_ne
            )
```

---

## 5. Serotonin (5-HT) -- Patience/Discounting

### Purpose

Serotonin encodes patience and long-horizon valuation. High 5-HT signals "be patient,
favor long-term associations." In computational terms, high 5-HT slows the decay of
eligibility traces, extending the temporal window over which credit can be assigned. Low
5-HT signals "prioritize immediate outcomes," causing traces to decay quickly and limiting
credit assignment to recent activity.

### Biological Analog

The dorsal raphe nucleus (DRN) is the primary source of serotonin in the brain. Serotonin
modulates temporal discounting: animals with depleted serotonin show increased impulsivity
and preference for smaller, sooner rewards over larger, later rewards. Optogenetic
activation of DRN serotonin neurons increases the willingness to wait for delayed rewards.

### Input Signals

| Signal | Shape | Description |
|--------|-------|-------------|
| `patience` | `(B,)` or `(B, 1)` | External patience signal or inverse impulsivity |
| `horizon_value` | `(B,)` or `(B, 1)` | Long-horizon value estimate uncertainty |
| `temporal_discount` | `(B,)` or `(B, 1)` | Current temporal discount factor gamma |

### Computation

```
raw_5ht = w_patience * patience + w_horizon * horizon_value + bias_5ht
5HT = sigmoid(raw_5ht)
```

where `w_patience` and `w_horizon` are learnable weights (initialized to 1.0) and
`bias_5ht` is a learnable bias (initialized to 0.0).

### Interaction with Eligibility Trace Decay

The central role of 5-HT is to modulate the effective time constant of eligibility traces.
Higher 5-HT extends the trace lifetime, allowing credit to be assigned to events further
in the past:

```
effective_tau_e = tau_e * (1 + alpha_5ht * 5HT)
```

where:
- `tau_e` is the base eligibility decay time constant from `EligibilityConfig`
- `alpha_5ht` is a scaling coefficient (default 1.0, configurable)
- `5HT` is the current serotonin level in [0, 1]

This means:
- When `5HT = 0`: `effective_tau_e = tau_e` (base decay, impatient)
- When `5HT = 0.5`: `effective_tau_e = 1.5 * tau_e` (50% longer traces)
- When `5HT = 1.0`: `effective_tau_e = 2.0 * tau_e` (doubled trace lifetime)

The decay factor per step becomes:

```
decay_factor = 1 - dt / effective_tau_e
```

where `dt` is the timestep duration.

### Output Range and Semantics

| 5-HT Value | Meaning | Effect on Credit Assignment |
|-------------|---------|---------------------------|
| 5-HT -> 1.0 | High patience | Slow trace decay, long-range credit assignment |
| 5-HT ~ 0.5 | Moderate patience | Baseline trace decay |
| 5-HT -> 0.0 | Low patience, impulsive | Fast trace decay, short-range credit only |

### Edge Cases

- **No patience signal available.** Set 5-HT = 0.5 (neutral). The trace decay runs at
  its baseline rate.

- **5-HT = 1.0 permanently.** The trace decay is halved (doubled time constant). This is
  valid for tasks that always require long-range credit assignment. Verify that traces do
  not accumulate unboundedly by ensuring the clamp range from `EligibilityConfig` is active.

- **alpha_5ht = 0.** Disables the 5-HT modulation of trace decay. The trace always decays
  at the base rate. Use this to ablate 5-HT effects in experiments.

### Pseudocode

```python
class SerotoninComputer(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_patience = nn.Parameter(torch.tensor(1.0))
        self.w_horizon = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        patience: Optional[torch.Tensor] = None,
        horizon_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute 5-HT from patience/horizon signals.

        Returns:
            sht: (B,) serotonin signal in [0, 1]
        """
        raw = self.bias.clone()
        if patience is not None:
            raw = raw + self.w_patience * patience.float()
        if horizon_value is not None:
            raw = raw + self.w_horizon * horizon_value.float()

        sht = torch.sigmoid(raw)
        return sht

    def modulate_tau(
        self,
        tau_e: float,
        sht: torch.Tensor,
        alpha_5ht: float = 1.0,
    ) -> torch.Tensor:
        """Compute effective trace decay time constant.

        Args:
            tau_e: Base eligibility time constant
            sht: (B,) serotonin level
            alpha_5ht: Scaling coefficient

        Returns:
            effective_tau: (B,) modulated time constant
        """
        return tau_e * (1.0 + alpha_5ht * sht)
```

---

## 6. Combination Functions

The four modulator outputs are combined into a single `global_plasticity` scalar that gates
the three-factor weight update. Three combination functions are supported, selected via
`NeuromodConfig.combination_fn`.

### 6.1 Weighted Sum

```
global_plasticity = w_da * DA + w_ach * ACh + w_ne * NE + w_5ht * 5HT
```

**Properties:**
- Simple and interpretable.
- Each weight controls how much influence a modulator has on overall plasticity.
- Default weights: `w_da = 0.4, w_ach = 0.3, w_ne = 0.2, w_5ht = 0.1`.
- The output is not naturally bounded. Apply a final clamp or tanh to bound the result.

**When to use:** Start with this for debugging and interpretability. Each modulator's
contribution is transparent and independently adjustable.

```python
def weighted_sum(da, ach, ne, sht, weights):
    """Weighted sum combination.

    Args:
        da: (B,) dopamine in [-1, 1]
        ach: (B,) acetylcholine in [0, 1]
        ne: (B,) norepinephrine in [0, 1]
        sht: (B,) serotonin in [0, 1]
        weights: (4,) learnable weights

    Returns:
        global_plasticity: (B,) plasticity gain
    """
    modulators = torch.stack([da, ach, ne, sht], dim=-1)  # (B, 4)
    raw = (modulators * weights).sum(dim=-1)               # (B,)
    return torch.tanh(raw)                                 # bound to [-1, 1]
```

### 6.2 Gated Product

```
global_plasticity = DA * (w_ach * ACh + (1 - w_ach) * baseline_gate)
```

**Properties:**
- DA is the primary driver: if DA = 0, plasticity is zero regardless of other modulators.
- ACh acts as a multiplicative gate on the DA signal: high ACh amplifies DA's effect.
- NE and 5-HT modulate secondary parameters (breadth and trace decay) rather than the
  primary plasticity gain.
- `baseline_gate` (default 0.5) prevents complete silence when ACh is low.

**When to use:** When reward-driven learning is the dominant paradigm and novelty should
amplify rather than independently trigger plasticity.

```python
def gated_product(da, ach, ne, sht, w_ach=0.7, baseline_gate=0.5):
    """Gated product combination.

    DA gates everything. ACh modulates the gate strength.

    Args:
        da: (B,) dopamine in [-1, 1]
        ach: (B,) acetylcholine in [0, 1]
        w_ach: weight for ACh gating (default 0.7)
        baseline_gate: minimum gate level (default 0.5)

    Returns:
        global_plasticity: (B,) plasticity gain in [-1, 1]
    """
    gate = w_ach * ach + (1 - w_ach) * baseline_gate  # (B,) in [baseline, 1]
    global_plasticity = da * gate                      # (B,) in [-1, 1]
    return global_plasticity
```

### 6.3 Learned MLP

```
global_plasticity = MLP([DA, ACh, NE, 5HT])
```

**Properties:**
- Most flexible: learns arbitrary nonlinear interactions between modulators.
- Output bounded by a final tanh activation.
- Requires more data to train the MLP weights.
- Less interpretable than weighted sum or gated product.

**Architecture:** A small two-layer MLP with hidden dimension from
`NeuromodConfig.modulator_hidden_dim` (default 64):

```
input (4) -> Linear(4, hidden) -> ReLU -> Linear(hidden, 1) -> tanh -> output (1)
```

**When to use:** When the interaction between modulators is complex and task-dependent,
and sufficient training data is available to learn the combination.

```python
class MLPCombination(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, da, ach, ne, sht):
        """Learned MLP combination.

        Args:
            da: (B,) dopamine in [-1, 1]
            ach: (B,) acetylcholine in [0, 1]
            ne: (B,) norepinephrine in [0, 1]
            sht: (B,) serotonin in [0, 1]

        Returns:
            global_plasticity: (B,) plasticity gain in [-1, 1]
        """
        modulators = torch.stack([da, ach, ne, sht], dim=-1)  # (B, 4)
        return self.mlp(modulators).squeeze(-1)                # (B,)
```

### Output Bounds Summary

All three combination functions produce `global_plasticity` in [-1, 1]:

| Function | Bounding Mechanism | Output Range |
|----------|-------------------|--------------|
| Weighted sum | Final `tanh` | [-1, 1] |
| Gated product | Product of DA in [-1,1] and gate in [0,1] | [-1, 1] |
| MLP | Final `tanh` layer | [-1, 1] |

### Selecting the Combination Function

Configure via `NeuromodConfig.combination_fn`:

```python
config = NeuromodConfig(
    combination_fn="weighted_sum",   # or "gated_product" or "mlp"
    modulator_hidden_dim=64,         # only used for "mlp"
)
```

---

## 7. State Management

`NeuromodulatoryGate` maintains running state that persists across forward calls within an
episode but resets between episodes (or on explicit reset).

### State Variables

| State Variable | Shape | Updated By | Purpose |
|----------------|-------|------------|---------|
| `reward_baseline` | `(1,)` | DA computation | EMA of recent rewards for RPE |
| `novelty_history` | `(history_len,)` | ACh computation | Rolling buffer for novelty normalization |
| `arousal_state` | `(1,)` | NE computation | Temporally smoothed NE for stable arousal |
| `history_idx` | `(1,)` integer | All | Current write index into circular buffers |

### State Operations

**`update_state(reward, novelty, urgency, patience)`**

Call this method once per timestep with the current raw signals. It updates all internal
state variables:

```python
def update_state(
    self,
    reward: Optional[torch.Tensor] = None,
    novelty: Optional[torch.Tensor] = None,
    urgency: Optional[torch.Tensor] = None,
    patience: Optional[torch.Tensor] = None,
):
    """Update internal state from current signals.

    Call once per timestep, after forward() has computed modulators.
    State updates are in-place and use no_grad.
    """
    with torch.no_grad():
        if reward is not None:
            # Update reward baseline (EMA)
            mean_reward = reward.detach().mean()
            self.reward_baseline.mul_(1 - self.alpha_baseline).add_(
                self.alpha_baseline * mean_reward
            )

        if novelty is not None:
            # Update novelty history (circular buffer)
            idx = self.history_idx.item() % self.history_len
            self.novelty_history[idx] = novelty.detach().mean()
            self.history_idx += 1

        if urgency is not None:
            # Update arousal state (temporal smoothing)
            mean_ne = urgency.detach().mean()
            self.arousal_state.mul_(1 - self.alpha_ne).add_(
                self.alpha_ne * mean_ne
            )
```

**`reset_state()`**

Reset all state to initial values. Call between episodes or when the task context changes
fundamentally:

```python
def reset_state(self):
    """Reset all internal state to initial values.

    Call between episodes or on task switch.
    """
    self.reward_baseline.zero_()
    self.novelty_history.zero_()
    self.arousal_state.fill_(0.5)
    self.history_idx.zero_()
```

### State Persistence Rules

1. **Within an episode:** State persists across timesteps. The reward baseline tracks the
   running mean, novelty history accumulates, and arousal smooths over time.

2. **Between episodes:** Call `reset_state()` to clear all state. Failing to reset between
   episodes causes cross-episode contamination -- the baseline from the previous episode
   biases DA computation in the new episode.

3. **Batch independence:** State variables store batch-averaged values (via `.mean()`). This
   means the state is shared across the batch for efficiency. Individual batch items still
   receive independent modulator values because the modulator computation uses per-item
   inputs; only the state update aggregates across the batch.

4. **Serialization:** State variables are registered as buffers (`register_buffer`), so they
   are included in `state_dict()` and restored by `load_state_dict()`. Checkpointing
   preserves the neuromodulatory state.

---

## 8. Determinism and Bounded Outputs

### Determinism Guarantees

All modulator computations are deterministic given the same inputs and internal state.
Verify this property with the following test:

```python
def test_modulator_determinism():
    gate = NeuromodulatoryGate(config)

    signals = {
        'reward': torch.tensor([0.5, -0.3, 1.0]),
        'novelty': torch.tensor([0.8, 0.2, 0.5]),
        'urgency': torch.tensor([0.1, 0.9, 0.4]),
        'patience': torch.tensor([0.6, 0.3, 0.7]),
    }

    results = []
    for _ in range(10):
        gate.reset_state()
        modulators, gp = gate(signals)
        results.append(gp.clone())

    for r in results[1:]:
        assert torch.equal(r, results[0]), "Modulator output is not deterministic"
```

There is no stochastic sampling, dropout, or random noise in the modulator pipeline.
During training, ensure that dropout layers (if any exist in the broader network) are not
applied within the modulator computation path.

### Bounded Output Guarantees

The following bounds hold by construction:

| Modulator | Activation | Guaranteed Range |
|-----------|-----------|-----------------|
| DA | `tanh` | [-1.0, 1.0] |
| ACh | `sigmoid` | [0.0, 1.0] |
| NE | `sigmoid` | [0.0, 1.0] |
| 5-HT | `sigmoid` | [0.0, 1.0] |
| `global_plasticity` | `tanh` or product | [-1.0, 1.0] |

These bounds are enforced by the activation functions themselves. Do not add explicit
clamp operations on top of tanh/sigmoid -- they are redundant and can mask gradient flow
issues.

The one exception is when external signals (e.g., `anomaly_score`) are added to the
post-activation modulator value. In the existing codebase, lines like
`ach = ach + 0.3 * anomaly_score` followed by `torch.clamp(ach, 0, 1)` exist. In the
skill-standardized implementation, prefer incorporating external signals before the
activation function (i.e., add them to the pre-activation `raw_ach`) so that the sigmoid
provides natural bounding without clamping.

### fp32 Requirements

Eligibility traces and modulator computations must run in fp32 for numerical stability.
The tanh and sigmoid functions have vanishing gradients at their extremes, and fp16
quantization can cause these gradients to become exactly zero, halting learning.

Enforce fp32 at the gate boundary:

```python
class NeuromodulatoryGate(nn.Module):
    @torch.cuda.amp.custom_fwd(cast_to=torch.float32)
    def forward(self, signals, state=None):
        # All internal computation happens in fp32
        ...
        return modulators, global_plasticity
```

Alternatively, wrap the forward call in `torch.autocast(enabled=False)`:

```python
with torch.autocast(device_type='cuda', enabled=False):
    modulators, gp = gate(signals_fp32)
```

---

## 9. Implementation Patterns

### 9.1 Complete NeuromodulatoryGate Class

The following pseudocode shows the full gate implementation integrating all four modulators
and the combination function:

```python
class NeuromodulatoryGate(nn.Module):
    """Computes DA/ACh/NE/5-HT and combines into global_plasticity.

    All computations are deterministic and bounded.
    State persists within episodes, resets between episodes.
    """

    def __init__(self, config: NeuromodConfig):
        super().__init__()
        self.config = config

        # Individual modulator computers
        self.da_computer = DopamineComputer(
            alpha_baseline=config.da_alpha_baseline,
        )
        self.ach_computer = AcetylcholineComputer(
            num_sources=config.ach_num_sources,
        )
        self.ne_computer = NorepinephrineComputer(
            alpha_smooth=config.ne_alpha_smooth,
        )
        self.sht_computer = SerotoninComputer()

        # Combination function
        if config.combination_fn == "weighted_sum":
            self.combination_weights = nn.Parameter(
                torch.tensor([0.4, 0.3, 0.2, 0.1])
            )
            self.combine = self._weighted_sum
        elif config.combination_fn == "gated_product":
            self.gate_weight = nn.Parameter(torch.tensor(0.7))
            self.combine = self._gated_product
        elif config.combination_fn == "mlp":
            self.combine_mlp = MLPCombination(config.modulator_hidden_dim)
            self.combine = self._mlp_combine
        else:
            raise ValueError(
                f"Unknown combination_fn: {config.combination_fn}"
            )

        # State buffers
        self.register_buffer('reward_baseline', torch.tensor(0.0))
        self.register_buffer(
            'novelty_history',
            torch.zeros(config.novelty_history_len),
        )
        self.register_buffer('arousal_state', torch.tensor(0.5))
        self.register_buffer(
            'history_idx', torch.tensor(0, dtype=torch.long)
        )

    @torch.cuda.amp.custom_fwd(cast_to=torch.float32)
    def forward(
        self,
        signals: Dict[str, torch.Tensor],
        state: Optional[Dict] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Compute all modulators and global plasticity gain.

        Args:
            signals: Dict with keys from {reward, novelty, entropy,
                     htm_anomaly, confidence, urgency, surprise,
                     patience, horizon_value}
            state: Optional external state override

        Returns:
            modulators: Dict {DA, ACh, NE, 5HT} each (B,)
            global_plasticity: (B,) combined plasticity gain
        """
        # --- Dopamine ---
        da = self.da_computer(
            reward=signals.get('reward'),
        )

        # --- Acetylcholine ---
        ach = self.ach_computer(
            novelty=signals.get('novelty'),
            entropy=signals.get('entropy'),
            htm_anomaly=signals.get('htm_anomaly'),
            confidence=signals.get('confidence'),
        )

        # --- Norepinephrine ---
        ne = self.ne_computer(
            urgency=signals.get('urgency'),
            surprise=signals.get('surprise'),
        )

        # --- Serotonin ---
        sht = self.sht_computer(
            patience=signals.get('patience'),
            horizon_value=signals.get('horizon_value'),
        )

        # --- Combine ---
        global_plasticity = self.combine(da, ach, ne, sht)

        modulators = {
            'DA': da,
            'ACh': ach,
            'NE': ne,
            '5HT': sht,
        }

        return modulators, global_plasticity

    def _weighted_sum(self, da, ach, ne, sht):
        mods = torch.stack([da, ach, ne, sht], dim=-1)
        raw = (mods * self.combination_weights).sum(dim=-1)
        return torch.tanh(raw)

    def _gated_product(self, da, ach, ne, sht):
        gate = self.gate_weight * ach + (1 - self.gate_weight) * 0.5
        return da * gate

    def _mlp_combine(self, da, ach, ne, sht):
        return self.combine_mlp(da, ach, ne, sht)

    def update_state(self, **signals):
        """Update running state from current signals."""
        if 'reward' in signals and signals['reward'] is not None:
            self.da_computer.update_baseline(signals['reward'])
        if 'ne_raw' in signals and signals['ne_raw'] is not None:
            self.ne_computer.update_smoothing(signals['ne_raw'])
        if 'novelty' in signals and signals['novelty'] is not None:
            with torch.no_grad():
                idx = (
                    self.history_idx.item() % len(self.novelty_history)
                )
                self.novelty_history[idx] = (
                    signals['novelty'].detach().mean()
                )
                self.history_idx += 1

    def reset_state(self):
        """Reset all internal state between episodes."""
        self.reward_baseline.zero_()
        self.novelty_history.zero_()
        self.arousal_state.fill_(0.5)
        self.history_idx.zero_()
        self.da_computer.baseline.zero_()
        self.ne_computer.ne_smooth.fill_(0.5)
```

### 9.2 Integration with EligibilityTraceModule

Connect the `NeuromodulatoryGate` to the `EligibilityTraceModule` via the three-factor
update rule. The gate produces `global_plasticity`, which serves as the third-factor
modulator signal:

```python
class ThreeFactorLearner(nn.Module):
    """Combines eligibility traces with neuromodulatory gating."""

    def __init__(self, config):
        super().__init__()
        self.trace_module = EligibilityTraceModule(config.eligibility)
        self.neuromod_gate = NeuromodulatoryGate(config.neuromod)
        self.lr = config.three_factor.lr

    def step(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        weights: torch.Tensor,
        signals: Dict[str, torch.Tensor],
        dt: float = 1.0,
    ) -> torch.Tensor:
        """One step of three-factor learning.

        Args:
            pre: (B, N_pre) pre-synaptic activations
            post: (B, N_post) post-synaptic activations
            weights: (N_post, N_pre) current weight matrix
            signals: modulator input signals
            dt: timestep duration

        Returns:
            updated_weights: (N_post, N_pre) updated weight matrix
        """
        # Step 1: Update eligibility traces from pre/post activity
        e = self.trace_module.update(pre, post, dt=dt)

        # Step 2: Compute neuromodulatory signals
        modulators, global_plasticity = self.neuromod_gate(signals)

        # Step 3: Modulate trace decay via 5-HT
        sht = modulators['5HT']
        effective_tau = self.neuromod_gate.sht_computer.modulate_tau(
            self.trace_module.config.tau_e, sht
        )
        self.trace_module.set_effective_tau(effective_tau)

        # Step 4: Compute and apply weight update
        # delta_w = lr * global_plasticity * e
        updated_weights = self.trace_module.apply_update(
            weights,
            mod_signal=global_plasticity,
            lr=self.lr,
        )

        # Step 5: Update neuromodulatory state
        self.neuromod_gate.update_state(**signals)

        return updated_weights
```

### 9.3 State Reset Pattern

Use the following pattern at episode boundaries:

```python
# At the start of each episode
three_factor_learner.trace_module.reset(batch_size, device)
three_factor_learner.neuromod_gate.reset_state()

# During the episode
for t in range(episode_length):
    # ... get pre, post, signals ...
    weights = three_factor_learner.step(
        pre, post, weights, signals, dt=1.0
    )

# At episode end: state is automatically ready for reset
```

### 9.4 Diagnostic Inspection

Extract modulator values for logging and debugging:

```python
# After forward pass
modulators, gp = gate(signals)

log_dict = {
    'neuromod/DA_mean': modulators['DA'].mean().item(),
    'neuromod/DA_std': modulators['DA'].std().item(),
    'neuromod/ACh_mean': modulators['ACh'].mean().item(),
    'neuromod/NE_mean': modulators['NE'].mean().item(),
    'neuromod/5HT_mean': modulators['5HT'].mean().item(),
    'neuromod/global_plasticity': gp.mean().item(),
    'neuromod/reward_baseline': gate.reward_baseline.item(),
    'neuromod/arousal_state': gate.arousal_state.item(),
}
# Send to tensorboard / wandb
```

Monitor for these warning signs:
- DA stuck at +1 or -1: the reward signal is too large or `w_reward` has grown too large.
- ACh always near 0 or always near 1: the novelty signal is not varying or weights have
  diverged.
- NE oscillating rapidly between 0 and 1: the temporal smoothing rate `alpha_ne` is too
  high.
- 5-HT always at 0.5: the patience signal is not being provided.
- `global_plasticity` always near zero: DA is dominating in gated product mode and the
  reward signal is absent.

---

## Summary of Formulas

| Modulator | Formula | Range |
|-----------|---------|-------|
| DA | `tanh(w_r * (reward - baseline))` | [-1, 1] |
| ACh | `sigmoid(w_n * novelty + w_e * entropy + b)` | [0, 1] |
| NE | `sigmoid(w_u * urgency + w_s * surprise + b)` | [0, 1] |
| 5-HT | `sigmoid(w_p * patience + w_h * horizon + b)` | [0, 1] |
| `global_plasticity` | `f(DA, ACh, NE, 5HT)` | [-1, 1] |
| `effective_tau_e` | `tau_e * (1 + alpha * 5HT)` | [tau_e, 2*tau_e] |

All weights (`w_*`) and biases (`b`) are learnable parameters. All activations (tanh,
sigmoid) provide natural bounding without explicit clamping.
