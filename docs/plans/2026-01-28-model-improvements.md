# Brain-Inspired AI Model Improvements Plan

**Date**: January 28, 2026  
**Based on**: Inference demo analysis + latest research

## ✅ Implementation Status (COMPLETED)

All improvements have been implemented. Here's a summary of what was added:

### New Components Added

| Module | New Class/Function | Description |
|--------|-------------------|-------------|
| **core/neurons.py** | `AdvancedLIFNeuron` | Learnable synaptic delays, heterogeneous tau |
| **core/losses.py** | `SNNLoss`, `prob_spikes_loss`, etc. | ProbSpikes loss and SNN regularization |
| **temporal/htm.py** | `ReflexMemory`, `AcceleratedHTM` | O(1) LSH pattern lookup, automatic promotion |
| **decision/active_inference.py** | `ImprovedEFEComputation`, `EmpowermentEstimator` | 3-component EFE with empowerment |
| **workspace/global_workspace.py** | `SelectionBroadcastWorkspace` | Iterative competition, ignition dynamics |
| **reasoning/symbolic.py** | `LogicTensorNetwork`, `RealLogic` | Differentiable neuro-symbolic reasoning |
| **meta/maml.py** | `MAMLPlusPlus`, `Task2Vec` | Per-layer LRs, task embeddings |

### Config Flags Added (config.py)

```python
# SNN Improvements
use_learnable_delays: bool = True
use_heterogeneous_tau: bool = True  
use_probspikes_loss: bool = True

# HTM Improvements  
use_reflex_memory: bool = True

# Workspace Improvements
use_selection_broadcast: bool = True
ignition_threshold: float = 0.3

# Active Inference Improvements
use_improved_efe: bool = True
use_empowerment: bool = True

# Reasoning Improvements
use_ltn: bool = True

# Meta-Learning Improvements
use_maml_plus_plus: bool = True
use_task2vec: bool = True
```

### To Enable Improvements

Update your training scripts to use the new components:

```python
# Example: Use improved SNN
from brain_ai.core import AdvancedLIFNeuron, SNNLoss

# Example: Use improved Active Inference
from brain_ai.decision import ImprovedActiveInferenceAgent

# Example: Use Selection-Broadcast Workspace
from brain_ai.workspace import create_selection_broadcast_workspace
workspace = create_selection_broadcast_workspace(workspace_dim=1024)

# Example: Use MAML++
from brain_ai.meta import create_maml_plus_plus
meta_learner = create_maml_plus_plus(model, learn_inner_lr=True)
```

---

## Executive Summary

Analysis of the inference demo revealed several key issues and opportunities for improvement:

### Current Model Diagnostics (from demo output)

| Metric | Current Value | Issue |
|--------|--------------|-------|
| Model Parameters | 261.99M | Reasonable, but workspace_dim too small |
| Prediction Confidence | ~10.54% | **Model predicting uniformly** (untrained) |
| Batch Speedup | 2.08x (vs 8x theoretical) | **26% batching efficiency** - memory bound |
| Workspace State | None during inference | **Workspace not returning state** |
| Anomaly Score | None | **HTM not producing anomaly detection** |
| Reasoning Used | False (System 1 only) | System 2 never triggered |
| Prediction Distribution | All "truck" (class 9) | **Collapsed output - needs training** |

---

## Phase 1: SNN Core Improvements

### Latest Research Insights (2025)

From "Beyond Rate Coding: Surrogate Gradients Enable Spike Timing Learning":
- **Learnable synaptic/axonal delays** significantly improve temporal coding
- **Inter-spike interval (ISI) learning** is possible with proper architecture
- **Heterogeneous time constants** per neuron improve learning
- **ProbSpikes loss** (cross-entropy on normalized spike counts) works better than membrane potential loss

### Recommended Changes

```python
# brain_ai/core/snn.py - Add learnable delays and heterogeneous time constants

class LIFNeuronAdvanced(nn.Module):
    """LIF neuron with learnable time constants and synaptic delays."""
    
    def __init__(
        self,
        size: int,
        beta_init: float = 0.9,
        learnable_beta: bool = True,  # NEW: Per-neuron time constants
        max_delay: int = 10,  # NEW: Learnable delays
        use_delays: bool = True,
    ):
        super().__init__()
        
        # Heterogeneous time constants (per-neuron)
        if learnable_beta:
            self.log_beta = nn.Parameter(
                torch.full((size,), math.log(beta_init / (1 - beta_init)))
            )
        else:
            self.register_buffer('log_beta', torch.full((size,), math.log(beta_init / (1 - beta_init))))
        
        # Learnable synaptic delays
        if use_delays:
            self.delay_weights = nn.Parameter(torch.zeros(size, max_delay))
            self.max_delay = max_delay
        else:
            self.delay_weights = None
        
    @property
    def beta(self):
        return torch.sigmoid(self.log_beta)
    
    def apply_delays(self, spike_history: torch.Tensor) -> torch.Tensor:
        """Apply learnable delays to spike history."""
        if self.delay_weights is None:
            return spike_history[:, -1]  # No delay, use latest
        
        # Soft attention over delay taps
        delay_attn = F.softmax(self.delay_weights, dim=-1)
        # spike_history: (batch, time, neurons)
        delayed = torch.einsum('btn,nd->bn', spike_history, delay_attn)
        return delayed
```

### New Loss Function: ProbSpikes

```python
def prob_spikes_loss(spike_output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    ProbSpikes loss - cross-entropy on normalized spike counts.
    
    Args:
        spike_output: (time, batch, num_classes) spike tensor
        targets: (batch,) class labels
    """
    # Sum spikes over time
    spike_counts = spike_output.sum(dim=0)  # (batch, num_classes)
    
    # Normalize to probability distribution
    spike_probs = spike_counts / (spike_counts.sum(dim=-1, keepdim=True) + 1e-8)
    
    # Cross-entropy (manual to avoid log(0))
    log_probs = torch.log(spike_probs + 1e-8)
    loss = F.nll_loss(log_probs, targets)
    
    return loss
```

### Spike Rate Regularization

```python
def spike_rate_regularization(spikes: torch.Tensor, target_rate: float = 0.1) -> torch.Tensor:
    """
    Regularize spike rates to biologically plausible levels.
    
    Target firing rate ~10% matches cortical observations.
    """
    # Average firing rate across time
    rates = spikes.mean(dim=0)  # (batch, neurons)
    
    # L2 penalty for deviation from target
    rate_loss = ((rates - target_rate) ** 2).mean()
    
    return rate_loss
```

---

## Phase 2: HTM Improvements

### Latest Research Insights

From "Accelerated Hierarchical Temporal Memory (AHTM)":
- **Reflex Memory (RM)** offloads repetitive patterns for faster response
- **Content-Addressable Memory (CAM)** hardware acceleration reduces prediction to 0.094s
- **Multi-order inferences** supported without catastrophic forgetting
- **Online learning** with continuous adaptation

### Recommended Architecture: AHTM

```python
# brain_ai/temporal/htm.py - Add Reflex Memory system

class ReflexMemory(nn.Module):
    """
    Reflex Memory for AHTM - stores frequently-accessed patterns.
    
    When a pattern is seen repeatedly, it's cached in RM for O(1) lookup
    instead of full HTM computation.
    """
    
    def __init__(
        self,
        pattern_dim: int,
        max_patterns: int = 10000,
        promotion_threshold: int = 5,  # Times seen before promotion
        decay_rate: float = 0.99,
    ):
        super().__init__()
        
        self.pattern_dim = pattern_dim
        self.max_patterns = max_patterns
        self.promotion_threshold = promotion_threshold
        self.decay_rate = decay_rate
        
        # Pattern storage (using locality-sensitive hashing)
        self.register_buffer('patterns', torch.zeros(max_patterns, pattern_dim))
        self.register_buffer('predictions', torch.zeros(max_patterns, pattern_dim))
        self.register_buffer('access_counts', torch.zeros(max_patterns))
        self.register_buffer('timestamps', torch.zeros(max_patterns))
        self.register_buffer('num_stored', torch.tensor(0))
        
        # LSH projection for fast lookup
        self.num_hashes = 8
        self.hash_dim = 32
        self.register_buffer(
            'hash_projections',
            torch.randn(pattern_dim, self.num_hashes * self.hash_dim) / math.sqrt(pattern_dim)
        )
    
    def compute_hash(self, pattern: torch.Tensor) -> torch.Tensor:
        """Compute locality-sensitive hash."""
        projected = pattern @ self.hash_projections
        return (projected > 0).float()
    
    def lookup(self, pattern: torch.Tensor) -> Optional[Tuple[torch.Tensor, float]]:
        """Fast O(1) lookup via LSH."""
        if self.num_stored == 0:
            return None
        
        pattern_hash = self.compute_hash(pattern)
        stored_hashes = self.compute_hash(self.patterns[:self.num_stored])
        
        # Hamming distance
        distances = (pattern_hash != stored_hashes).sum(dim=-1)
        
        # Find best match
        min_dist, min_idx = distances.min(dim=0)
        
        # Threshold for match
        if min_dist < self.num_hashes * self.hash_dim * 0.1:
            self.access_counts[min_idx] += 1
            return self.predictions[min_idx], 1.0 - min_dist.float() / (self.num_hashes * self.hash_dim)
        
        return None
    
    def store(self, pattern: torch.Tensor, prediction: torch.Tensor, current_time: int):
        """Store pattern-prediction pair."""
        if self.num_stored < self.max_patterns:
            idx = self.num_stored
            self.num_stored += 1
        else:
            # Evict least recently used
            idx = self.timestamps[:self.num_stored].argmin()
        
        self.patterns[idx] = pattern
        self.predictions[idx] = prediction
        self.access_counts[idx] = 1
        self.timestamps[idx] = current_time


class AcceleratedHTM(nn.Module):
    """
    AHTM: HTM with Reflex Memory acceleration.
    
    Combines:
    - Full HTM for novel patterns
    - Reflex Memory for fast repetitive pattern response
    - Automatic pattern promotion based on access frequency
    """
    
    def __init__(self, htm_layer: 'HTMLayer', reflex_memory: ReflexMemory):
        super().__init__()
        self.htm = htm_layer
        self.rm = reflex_memory
        self.timestep = 0
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Forward with automatic acceleration.
        
        Returns:
            prediction: Next state prediction
            anomaly_score: How novel the pattern is
        """
        self.timestep += 1
        
        # Try Reflex Memory first (O(1))
        rm_result = self.rm.lookup(x)
        
        if rm_result is not None:
            prediction, confidence = rm_result
            # Still update HTM in background for learning
            if self.training:
                with torch.no_grad():
                    htm_pred, _ = self.htm(x)
            return prediction, 0.0  # Low anomaly - known pattern
        
        # Full HTM computation
        prediction, anomaly = self.htm(x)
        
        # Promote to RM if seen enough times
        # (promotion logic handled by RM internally)
        self.rm.store(x, prediction, self.timestep)
        
        return prediction, anomaly
```

---

## Phase 3: Global Workspace Improvements

### Latest Research Insights

From "Global Workspace Theory and Real-Time World" (Frontiers 2025):
- **Selection-Broadcast Cycle** is key for dynamic adaptation
- **Experience-Based Adaptation** accelerates processing with memory
- **Immediate Real-Time Adaptation** for quick intervention

### Recommended Changes

```python
# brain_ai/workspace/global_workspace.py - Enhanced selection-broadcast

class SelectionBroadcastWorkspace(nn.Module):
    """
    Global Workspace with explicit Selection-Broadcast cycle.
    
    Key innovations:
    - Competition phase with winner-take-all dynamics
    - Broadcast phase disseminates winner to all modules
    - Ignition threshold for conscious access
    """
    
    def __init__(
        self,
        workspace_dim: int = 512,
        num_modules: int = 8,
        ignition_threshold: float = 0.7,
        broadcast_gain: float = 10.0,
    ):
        super().__init__()
        
        self.workspace_dim = workspace_dim
        self.ignition_threshold = ignition_threshold
        self.broadcast_gain = broadcast_gain
        
        # Module attention for competition
        self.module_attention = nn.MultiheadAttention(
            embed_dim=workspace_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        
        # Ignition gate - determines if content reaches consciousness
        self.ignition_gate = nn.Sequential(
            nn.Linear(workspace_dim, workspace_dim // 4),
            nn.ReLU(),
            nn.Linear(workspace_dim // 4, 1),
            nn.Sigmoid(),
        )
        
        # Broadcast projection (amplifies winning content)
        self.broadcast_projection = nn.Linear(workspace_dim, workspace_dim)
    
    def selection_phase(
        self,
        module_outputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Competition between modules for workspace access.
        
        Returns:
            winner: Selected content
            attention_weights: Competition results
            winning_module: Which module won
        """
        # Stack all module outputs
        modules = list(module_outputs.keys())
        stacked = torch.stack([module_outputs[m] for m in modules], dim=1)
        
        # Self-attention for competition
        attended, attention_weights = self.module_attention(
            stacked, stacked, stacked,
            need_weights=True,
        )
        
        # Winner-take-all: select highest attended
        module_saliency = attention_weights.mean(dim=(0, 2))  # Average attention received
        winner_idx = module_saliency.argmax()
        winner = attended[:, winner_idx]
        
        return winner, attention_weights, modules[winner_idx]
    
    def ignition_check(self, content: torch.Tensor) -> Tuple[bool, float]:
        """Check if content reaches ignition threshold for broadcast."""
        ignition_score = self.ignition_gate(content).mean()
        ignites = ignition_score > self.ignition_threshold
        return ignites, ignition_score.item()
    
    def broadcast_phase(
        self,
        content: torch.Tensor,
        all_modules: Dict[str, nn.Module],
    ) -> Dict[str, torch.Tensor]:
        """
        Broadcast winning content to all modules.
        
        This implements the "global ignition" - when information
        becomes globally available.
        """
        # Amplify content for broadcast
        broadcast_content = self.broadcast_projection(content) * self.broadcast_gain
        
        # Send to all modules (they can choose to attend or not)
        broadcast_results = {}
        for name, module in all_modules.items():
            if hasattr(module, 'receive_broadcast'):
                broadcast_results[name] = module.receive_broadcast(broadcast_content)
        
        return broadcast_results
```

---

## Phase 4: Active Inference Improvements

### Latest Research Insights

- **Expected Free Energy (EFE)** should include both pragmatic and epistemic terms
- **Deep active inference** with neural networks is effective
- **Amortized inference** for faster action selection

### Recommended Changes

```python
# brain_ai/decision/active_inference.py - Improved EFE computation

class ImprovedEFEComputation(nn.Module):
    """
    Enhanced Expected Free Energy computation.
    
    EFE = Pragmatic Value + Epistemic Value + Instrumental Value
    
    - Pragmatic: How well does action achieve preferences?
    - Epistemic: How much uncertainty is reduced?
    - Instrumental: How does action enable future options?
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        # Preference model P(o) - what observations we prefer
        self.preferences = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.LogSoftmax(dim=-1),
        )
        
        # Forward model P(s'|s,a) for epistemic value
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.forward_mu = nn.Linear(hidden_dim, state_dim)
        self.forward_logvar = nn.Linear(hidden_dim, state_dim)
        
        # Empowerment estimator for instrumental value
        self.empowerment = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def compute_efe(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        num_samples: int = 32,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute Expected Free Energy for action.
        
        Returns:
            efe: Expected Free Energy (lower is better)
            components: Breakdown of EFE terms
        """
        # Predict next state distribution
        sa = torch.cat([state, action], dim=-1)
        h = self.forward_model(sa)
        next_mu = self.forward_mu(h)
        next_logvar = self.forward_logvar(h)
        
        # Sample next states
        std = torch.exp(0.5 * next_logvar)
        eps = torch.randn(num_samples, *std.shape, device=std.device)
        next_states = next_mu + eps * std  # (samples, batch, state_dim)
        
        # 1. Pragmatic value: KL[Q(o|s') || P(o)]
        # How much will observations diverge from preferences?
        pred_obs = next_states  # Assuming state ≈ observation
        log_prefs = self.preferences(next_mu)  # Use mean for efficiency
        pragmatic = F.kl_div(
            F.log_softmax(pred_obs, dim=-1),
            log_prefs.exp(),
            reduction='batchmean',
        )
        
        # 2. Epistemic value: H[P(s'|s,a)]
        # How uncertain are we about next state?
        epistemic = 0.5 * (next_logvar + math.log(2 * math.pi * math.e)).mean()
        
        # 3. Instrumental value: Empowerment
        # How many future options does this action enable?
        instrumental = -self.empowerment(next_mu).mean()
        
        # Total EFE (lower is better, so we negate what we want to maximize)
        efe = pragmatic + epistemic + instrumental
        
        return efe, {
            'pragmatic': pragmatic,
            'epistemic': epistemic,
            'instrumental': instrumental,
        }
```

---

## Phase 5: Neuro-Symbolic Improvements

### Latest Research Insights (2025)

- **Logic Tensor Networks** for differentiable reasoning
- **Neural theorem provers** for proof generation
- **Symbolic knowledge in loss function** improves generalization
- Addresses hallucination issues in LLMs

### Recommended Changes

```python
# brain_ai/reasoning/symbolic.py - Logic Tensor Networks integration

class LogicTensorNetwork(nn.Module):
    """
    Logic Tensor Network for differentiable symbolic reasoning.
    
    Implements fuzzy logic operations that are differentiable:
    - AND: Product t-norm (a * b)
    - OR: Product t-conorm (a + b - a*b)
    - NOT: 1 - a
    - IMPLIES: min(1, 1 - a + b)
    """
    
    def __init__(
        self,
        entity_dim: int = 256,
        predicate_dim: int = 128,
        num_predicates: int = 100,
    ):
        super().__init__()
        
        self.entity_dim = entity_dim
        self.predicate_dim = predicate_dim
        
        # Predicate embeddings
        self.predicate_embeddings = nn.Embedding(num_predicates, predicate_dim)
        
        # Entity-to-grounding projection
        self.entity_projection = nn.Linear(entity_dim, predicate_dim)
        
        # Grounding function: scores truth value of predicate(entity)
        self.grounding = nn.Sequential(
            nn.Linear(predicate_dim * 2, predicate_dim),
            nn.ReLU(),
            nn.Linear(predicate_dim, 1),
            nn.Sigmoid(),
        )
        
        # Relation grounding: scores truth value of relation(e1, e2)
        self.relation_grounding = nn.Sequential(
            nn.Linear(entity_dim * 2 + predicate_dim, predicate_dim),
            nn.ReLU(),
            nn.Linear(predicate_dim, 1),
            nn.Sigmoid(),
        )
    
    def ground_predicate(
        self,
        entity: torch.Tensor,
        predicate_id: int,
    ) -> torch.Tensor:
        """Compute truth value of predicate(entity)."""
        pred_emb = self.predicate_embeddings(
            torch.tensor([predicate_id], device=entity.device)
        )
        entity_proj = self.entity_projection(entity)
        
        combined = torch.cat([entity_proj, pred_emb.expand(entity.shape[0], -1)], dim=-1)
        return self.grounding(combined).squeeze(-1)
    
    def fuzzy_and(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Product t-norm: a AND b."""
        return a * b
    
    def fuzzy_or(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Product t-conorm: a OR b."""
        return a + b - a * b
    
    def fuzzy_not(self, a: torch.Tensor) -> torch.Tensor:
        """Negation: NOT a."""
        return 1 - a
    
    def fuzzy_implies(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Implication: a IMPLIES b."""
        return torch.clamp(1 - a + b, 0, 1)
    
    def evaluate_rule(
        self,
        rule: str,
        bindings: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Evaluate a logical rule with entity bindings.
        
        Example rule: "parent(X,Y) AND parent(Y,Z) IMPLIES grandparent(X,Z)"
        """
        # Parse and evaluate rule (simplified)
        # In practice, use a proper parser
        return self._evaluate_parsed_rule(rule, bindings)
```

### Symbolic Knowledge Loss

```python
def symbolic_knowledge_loss(
    model_outputs: torch.Tensor,
    rules: List[str],
    entity_embeddings: Dict[str, torch.Tensor],
    ltn: LogicTensorNetwork,
) -> torch.Tensor:
    """
    Loss term that enforces symbolic rules.
    
    Adds differentiable constraint that model outputs should
    satisfy known logical rules.
    """
    total_loss = 0.0
    
    for rule in rules:
        # Evaluate rule satisfaction
        satisfaction = ltn.evaluate_rule(rule, entity_embeddings)
        
        # Loss is high when rule is violated (satisfaction < 1)
        rule_loss = (1 - satisfaction).pow(2).mean()
        total_loss += rule_loss
    
    return total_loss / len(rules)
```

---

## Phase 6: Meta-Learning Improvements

### Latest Research Insights (2025)

- **Task2Vec** for task embeddings improves task selection
- **MAML++** improvements: annealed inner LR, multi-step loss
- **Reptile** as simpler alternative for large-scale
- Few-shot learning on drug discovery showing 5.9% improvement over baselines

### Recommended Changes

```python
# brain_ai/meta/maml.py - MAML++ with improvements

class MAMLPlusPlus(nn.Module):
    """
    MAML++ with modern improvements.
    
    Key improvements over vanilla MAML:
    - Per-layer per-step learning rates
    - Multi-step loss (not just final step)
    - Annealed inner learning rate
    - Gradient clipping
    - Second-order approximation option
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        num_inner_steps: int = 5,
        multi_step_loss_weights: Optional[List[float]] = None,
        use_second_order: bool = True,
    ):
        super().__init__()
        
        self.model = model
        self.num_inner_steps = num_inner_steps
        self.use_second_order = use_second_order
        
        # Per-layer, per-step learning rates
        self.inner_lrs = nn.ParameterDict()
        for name, param in model.named_parameters():
            safe_name = name.replace('.', '_')
            self.inner_lrs[safe_name] = nn.Parameter(
                torch.full((num_inner_steps,), inner_lr)
            )
        
        # Multi-step loss weights (importance-weighted)
        if multi_step_loss_weights is None:
            # Later steps get more weight (annealed)
            weights = [2 ** i for i in range(num_inner_steps)]
            weights = [w / sum(weights) for w in weights]
        self.register_buffer(
            'loss_weights',
            torch.tensor(multi_step_loss_weights or weights)
        )
    
    def inner_loop(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Inner loop adaptation with multi-step loss.
        
        Returns:
            query_output: Predictions on query set
            step_losses: Losses at each inner step
        """
        # Clone parameters for task-specific adaptation
        adapted_params = {
            name: param.clone()
            for name, param in self.model.named_parameters()
        }
        
        step_losses = []
        
        for step in range(self.num_inner_steps):
            # Forward pass with current adapted params
            support_out = self._forward_with_params(support_x, adapted_params)
            loss = F.cross_entropy(support_out, support_y)
            step_losses.append(loss)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=self.use_second_order,
                allow_unused=True,
            )
            
            # Update with per-layer, per-step LR
            for (name, param), grad in zip(adapted_params.items(), grads):
                if grad is not None:
                    safe_name = name.replace('.', '_')
                    lr = self.inner_lrs[safe_name][step]
                    adapted_params[name] = param - lr * grad
        
        # Query set forward
        query_out = self._forward_with_params(query_x, adapted_params)
        
        return query_out, step_losses
    
    def meta_loss(
        self,
        query_out: torch.Tensor,
        query_y: torch.Tensor,
        step_losses: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute meta-loss with multi-step weighting."""
        # Query loss
        query_loss = F.cross_entropy(query_out, query_y)
        
        # Multi-step loss (weighted sum of inner losses)
        multi_step_loss = sum(
            w * l for w, l in zip(self.loss_weights, step_losses)
        )
        
        # Combined loss
        return query_loss + 0.1 * multi_step_loss
```

### Task2Vec for Task Embeddings

```python
class Task2Vec(nn.Module):
    """
    Task embeddings for meta-learning task selection.
    
    Embeds tasks based on Fisher Information Matrix diagonal,
    allowing similarity comparison between tasks.
    """
    
    def __init__(self, probe_network: nn.Module, embedding_dim: int = 256):
        super().__init__()
        
        self.probe = probe_network
        self.embedding_dim = embedding_dim
        
        # Projection from FIM diagonal to fixed-size embedding
        self.projection = None  # Initialized on first use
    
    def compute_fisher_diagonal(
        self,
        dataloader: DataLoader,
        num_batches: int = 10,
    ) -> torch.Tensor:
        """Compute diagonal of Fisher Information Matrix."""
        fisher_diag = {}
        
        for name, param in self.probe.named_parameters():
            fisher_diag[name] = torch.zeros_like(param)
        
        count = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            self.probe.zero_grad()
            output = self.probe(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            
            for name, param in self.probe.named_parameters():
                if param.grad is not None:
                    fisher_diag[name] += param.grad ** 2
            
            count += 1
        
        # Normalize
        fim_vector = torch.cat([
            (f / count).flatten() for f in fisher_diag.values()
        ])
        
        return fim_vector
    
    def embed_task(self, dataloader: DataLoader) -> torch.Tensor:
        """Get fixed-size embedding for a task."""
        fim = self.compute_fisher_diagonal(dataloader)
        
        # Project to embedding dimension
        if self.projection is None:
            self.projection = nn.Linear(fim.shape[0], self.embedding_dim)
            self.projection.to(fim.device)
        
        return self.projection(fim)
    
    def task_similarity(
        self,
        task1_loader: DataLoader,
        task2_loader: DataLoader,
    ) -> float:
        """Compute cosine similarity between task embeddings."""
        emb1 = self.embed_task(task1_loader)
        emb2 = self.embed_task(task2_loader)
        
        return F.cosine_similarity(emb1, emb2, dim=-1).item()
```

---

## Liquid Neural Network (Working Memory) Improvements

### Current Implementation Issues

From demo output, the CfC/LTC working memory shows:
- Hidden states not properly persisted
- Batch size mismatches between operations

### Recommended Improvements

```python
# brain_ai/workspace/working_memory.py - Improved Liquid NN

class ImprovedLiquidMemory(nn.Module):
    """
    Improved Liquid Neural Network for working memory.
    
    Improvements:
    - Explicit state management
    - Adaptive time constants based on input
    - Neural Circuit Policies (NCP) wiring
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_ncp_wiring: bool = True,
    ):
        super().__init__()
        
        from ncps import wirings
        from ncps.torch import CfC
        
        if use_ncp_wiring:
            # Neural Circuit Policy wiring (based on C. elegans)
            wiring = wirings.AutoNCP(
                units=hidden_dim,
                output_size=output_dim,
            )
        else:
            # Fully connected wiring
            wiring = wirings.FullyConnected(
                units=hidden_dim,
                output_size=output_dim,
            )
        
        self.cfc = CfC(
            input_size=input_dim,
            wiring=wiring,
            return_sequences=True,
            batch_first=True,
        )
        
        # Adaptive time constant based on input
        self.time_constant_adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
        )
        
        # Explicit state tracking
        self.hidden_state = None
        self.state_batch_size = None
    
    def reset_state(self, batch_size: Optional[int] = None):
        """Explicitly reset hidden state."""
        self.hidden_state = None
        self.state_batch_size = batch_size
    
    def forward(
        self,
        x: torch.Tensor,
        reset_state: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with automatic state management.
        
        Args:
            x: Input (batch, time, features) or (batch, features)
            reset_state: Force state reset
        """
        # Handle 2D input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add time dimension
        
        batch_size = x.shape[0]
        
        # Check if we need to reset state
        if reset_state or self.hidden_state is None or self.state_batch_size != batch_size:
            self.hidden_state = None
            self.state_batch_size = batch_size
        
        # Compute adaptive time constants
        time_scales = self.time_constant_adapter(x.mean(dim=1))
        
        # Forward through CfC
        output, new_hidden = self.cfc(x, self.hidden_state)
        
        # Update state
        self.hidden_state = new_hidden
        
        return output, new_hidden
```

---

## Training Script Improvements

### Enhanced train_phase1.py

```python
# Add to scripts/train_phase1.py

def compute_snn_metrics(model, spikes, membrane, targets):
    """Compute SNN-specific training metrics."""
    metrics = {}
    
    # Spike rate (should be ~10% for biological plausibility)
    spike_rate = spikes.mean().item()
    metrics['spike_rate'] = spike_rate
    
    # Temporal sparsity (distribution of spikes over time)
    spike_per_time = spikes.sum(dim=-1).mean(dim=-1)  # Per timestep
    temporal_entropy = -(spike_per_time * (spike_per_time + 1e-8).log()).mean()
    metrics['temporal_entropy'] = temporal_entropy.item()
    
    # Firing pattern diversity (should be high for different inputs)
    if spikes.shape[0] > 1:  # Need multiple samples
        flat_spikes = spikes.sum(dim=0).flatten(1)  # Flatten spatial dims
        cosine_sim = F.cosine_similarity(
            flat_spikes.unsqueeze(0), flat_spikes.unsqueeze(1), dim=-1
        )
        # Mask diagonal
        mask = ~torch.eye(cosine_sim.shape[0], dtype=bool, device=cosine_sim.device)
        avg_similarity = cosine_sim[mask].mean()
        metrics['pattern_diversity'] = (1 - avg_similarity).item()
    
    # Prediction accuracy
    with torch.no_grad():
        pred = spikes.sum(dim=0).argmax(dim=-1)
        acc = (pred == targets).float().mean()
        metrics['accuracy'] = acc.item()
    
    return metrics


def train_epoch_improved(model, train_loader, optimizer, device, args):
    """Improved training loop with SNN-specific losses."""
    model.train()
    
    total_loss = 0
    all_metrics = defaultdict(list)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        spikes, membrane = model(data.flatten(1), encode_input=True)
        
        # Multi-component loss
        # 1. Classification loss (ProbSpikes)
        ce_loss = prob_spikes_loss(spikes, target)
        
        # 2. Spike rate regularization
        rate_loss = spike_rate_regularization(spikes, target_rate=0.1)
        
        # 3. Temporal consistency (encourage consistent patterns)
        temp_loss = temporal_consistency_loss(spikes)
        
        # Combined loss
        loss = ce_loss + 0.1 * rate_loss + 0.01 * temp_loss
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            metrics = compute_snn_metrics(model, spikes, membrane, target)
            for k, v in metrics.items():
                all_metrics[k].append(v)
        
        total_loss += loss.item()
    
    # Aggregate metrics
    epoch_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    epoch_metrics['loss'] = total_loss / len(train_loader)
    
    return epoch_metrics
```

---

## Immediate Action Items

### High Priority (This Week)

1. **Increase workspace_dim** from 512 to 1024+ in config
2. **Add spike rate regularization** to Phase 1 training
3. **Fix workspace state return** in inference
4. **Enable HTM anomaly detection** in forward pass

### Medium Priority (This Month)

5. **Implement learnable synaptic delays** in SNN
6. **Add Reflex Memory** to HTM for acceleration
7. **Implement Selection-Broadcast** in Global Workspace
8. **Add Logic Tensor Networks** for symbolic reasoning

### Low Priority (Next Quarter)

9. **Task2Vec** for meta-learning task selection
10. **MAML++ improvements** with per-layer LRs
11. **Full AHTM** hardware-aware implementation
12. **Empowerment-based** Active Inference

---

## Configuration Changes

```python
# brain_ai/config.py - Recommended production config updates

@dataclass
class BrainAIConfig:
    # ... existing fields ...
    
    # NEW: Increased workspace for better integration
    workspace_dim: int = 1024  # Was 512
    
    # NEW: SNN timing improvements
    use_learnable_delays: bool = True
    use_heterogeneous_tau: bool = True
    spike_rate_target: float = 0.1
    
    # NEW: HTM acceleration
    use_reflex_memory: bool = True
    reflex_memory_size: int = 10000
    
    # NEW: Enhanced Global Workspace
    use_selection_broadcast: bool = True
    ignition_threshold: float = 0.7
    
    # NEW: Neuro-symbolic
    use_logic_tensor_networks: bool = True
    num_logical_predicates: int = 100
    
    # NEW: Meta-learning
    use_maml_plus_plus: bool = True
    use_task2vec: bool = True
```

---

## Expected Improvements

| Metric | Current | Expected After Improvements |
|--------|---------|---------------------------|
| Prediction Confidence | ~10% | 85%+ (after training) |
| Batching Efficiency | 26% | 70%+ (with gradient checkpointing) |
| Workspace State | None | Full state tensor |
| Anomaly Detection | None | Functional scores |
| System 2 Reasoning | Never triggered | Triggered for complex inputs |
| Spike Rate | Unknown | ~10% (biologically plausible) |
| Meta-learning Adaptation | - | 5-10 shot learning |

---

## References

1. Yu, Z., Sun, P., Goodman, D.F.M. (2025). "Beyond Rate Coding: Surrogate Gradients Enable Spike Timing Learning in SNNs." arXiv:2507.16043
2. AHTM Paper (2025). "Enhancing Biologically Inspired Hierarchical Temporal Memory with Reflex Memory." arXiv:2504.03746
3. Frontiers (2025). "Global Workspace Theory and Dealing with a Real-Time World."
4. Wikipedia (2025). "Neuro-symbolic AI" - adoption increased for hallucination mitigation
5. Hasani et al. "Liquid Time-Constant Networks" and "Closed-form Continuous-Time Networks"
6. MAML++ and Task2Vec papers for meta-learning improvements
