"""
Brain-Inspired AI with Engram Conditional Memory
================================================

This module integrates DeepSeek's Engram (conditional memory) with a 
brain-inspired architecture, creating a system with multiple memory types:

1. **Engram (Semantic Memory)**: O(1) lookup for static patterns
   - Like the brain's semantic memory system
   - Fast retrieval of learned associations
   - Complements deep computation

2. **HTM (Episodic/Sequential Memory)**: Online sequence learning
   - Learns temporal patterns without forgetting
   - Anomaly detection via prediction failure

3. **Global Workspace (Working Memory)**: Attention-gated integration
   - Capacity-limited (Miller's 7±2)
   - Information broadcast to all modules

4. **SNN Core (Neural Computation)**: Sparse temporal processing
   - Event-driven, energy-efficient
   - Temporal coding enables precise timing

The key insight from Engram: language processing involves both:
- Compositional reasoning (needs computation) → SNN/Transformer
- Pattern retrieval (needs memory) → Engram

This maps to dual-process theory (System 1/System 2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import math

# Import Engram components
from engram_memory import (
    EngramConfig, 
    EngramModule, 
    EngramEmbedding,
    ContextAwareGating,
    RMSNorm
)


@dataclass
class BrainInspiredConfig:
    """Configuration for brain-inspired AI with Engram."""
    
    # Input/Output
    vocab_size: int = 50257
    hidden_dim: int = 512
    output_dim: int = 10
    
    # Engram (Semantic Memory)
    use_engram: bool = True
    engram_dim: int = 256
    engram_ngrams: List[int] = None  # Default: [2, 3]
    engram_heads: int = 8
    engram_layers: List[int] = None  # Which layers get Engram
    
    # SNN (Neural Computation)
    use_snn: bool = True
    snn_timesteps: int = 16
    snn_beta: float = 0.9  # Membrane decay
    
    # Working Memory (Global Workspace)
    use_workspace: bool = True
    workspace_dim: int = 256
    workspace_heads: int = 8
    capacity_limit: int = 7  # Miller's Law
    
    # Architecture
    num_layers: int = 6
    num_heads: int = 8
    ffn_multiplier: int = 4
    dropout: float = 0.1
    
    def __post_init__(self):
        if self.engram_ngrams is None:
            self.engram_ngrams = [2, 3]
        if self.engram_layers is None:
            # Early layer for local pattern offloading
            self.engram_layers = [1]


# =============================================================================
# Spiking Neural Network Components
# =============================================================================

class SpikingNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) neuron with surrogate gradients.
    
    Dynamics: U[t+1] = β·U[t] + W·X[t] - S[t]·V_thresh
    Spike: S[t] = 1 if U[t] > V_thresh else 0
    
    Uses arctangent surrogate gradient for backpropagation.
    """
    
    def __init__(
        self,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_slope: float = 25.0
    ):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.surrogate_slope = surrogate_slope
    
    def forward(
        self,
        x: torch.Tensor,
        mem: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single timestep forward pass.
        
        Args:
            x: Input current (batch, features)
            mem: Previous membrane potential
            
        Returns:
            spike: Binary spike output
            mem: Updated membrane potential
        """
        if mem is None:
            mem = torch.zeros_like(x)
        
        # Integrate
        mem = self.beta * mem + x
        
        # Spike with surrogate gradient
        spike = self.spike_fn(mem - self.threshold)
        
        # Reset
        mem = mem - spike * self.threshold
        
        return spike, mem
    
    def spike_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Spike function with arctangent surrogate gradient."""
        # Forward: Heaviside step function
        spike = (x > 0).float()
        
        # Backward: Surrogate gradient (arctangent)
        if self.training:
            surrogate = 1.0 / (1.0 + (math.pi * self.surrogate_slope * x) ** 2)
            spike = spike - spike.detach() + spike.detach() * surrogate
        
        return spike


class SNNLayer(nn.Module):
    """
    Spiking Neural Network layer with temporal processing.
    
    Processes input over T timesteps, accumulating spikes for rate coding.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_timesteps: int = 16,
        beta: float = 0.9
    ):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        self.linear = nn.Linear(in_features, out_features)
        self.neuron = SpikingNeuron(beta=beta)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input over timesteps.
        
        Args:
            x: (batch, features) or (batch, time, features)
            
        Returns:
            spike_sum: Rate-coded output (batch, out_features)
            spike_record: All spikes (time, batch, out_features)
        """
        if x.dim() == 2:
            # Static input: repeat for each timestep
            x = x.unsqueeze(0).repeat(self.num_timesteps, 1, 1)
        else:
            x = x.transpose(0, 1)  # (time, batch, features)
        
        spike_record = []
        mem = None
        
        for t in range(x.shape[0]):
            current = self.linear(x[t])
            spike, mem = self.neuron(current, mem)
            spike_record.append(spike)
        
        spike_record = torch.stack(spike_record)  # (time, batch, features)
        spike_sum = spike_record.sum(dim=0)  # (batch, features)
        
        return spike_sum, spike_record


# =============================================================================
# Global Workspace (Working Memory)
# =============================================================================

class GlobalWorkspace(nn.Module):
    """
    Global Workspace for information integration.
    
    Implements "ignition" from Global Workspace Theory:
    - Information from specialist modules competes for access
    - Winners broadcast globally to all modules
    - Capacity-limited (7±2 items)
    
    The workspace integrates information from:
    - Engram (semantic memory)
    - SNN (neural processing)
    - Other sensory/cognitive modules
    """
    
    def __init__(
        self,
        workspace_dim: int,
        num_heads: int = 8,
        capacity_limit: int = 7
    ):
        super().__init__()
        
        self.workspace_dim = workspace_dim
        self.capacity_limit = capacity_limit
        
        # Competition mechanism
        self.competition = nn.MultiheadAttention(
            embed_dim=workspace_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Workspace state evolution (Liquid-like dynamics)
        self.state_update = nn.GRUCell(workspace_dim, workspace_dim)
        
        # Capacity gate
        self.capacity_gate = nn.Sequential(
            nn.Linear(workspace_dim, workspace_dim // 4),
            nn.ReLU(),
            nn.Linear(workspace_dim // 4, 1)
        )
        
        self.norm = RMSNorm(workspace_dim)
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        workspace_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Integrate information from multiple modules.
        
        Args:
            inputs: Dict of named inputs {module_name: (batch, features)}
            workspace_state: Previous workspace state
            
        Returns:
            integrated: (batch, workspace_dim) integrated representation
            attention: Attention weights showing what "won" competition
            new_state: Updated workspace state
        """
        batch_size = list(inputs.values())[0].shape[0]
        device = list(inputs.values())[0].device
        
        # Stack all inputs
        stacked = torch.stack(list(inputs.values()), dim=1)  # (batch, num_inputs, dim)
        
        # Initialize workspace state if needed
        if workspace_state is None:
            workspace_state = stacked.mean(dim=1)  # (batch, dim)
        
        # Competition via cross-attention
        query = workspace_state.unsqueeze(1)  # (batch, 1, dim)
        attended, attention = self.competition(
            query, stacked, stacked
        )  # attended: (batch, 1, dim)
        
        # Capacity constraint via soft top-k
        gate_scores = self.capacity_gate(stacked).squeeze(-1)  # (batch, num_inputs)
        gate_probs = F.softmax(gate_scores / 0.1, dim=-1)  # Sharpened softmax
        
        # Top-k selection
        topk_probs, topk_idx = torch.topk(gate_probs, min(self.capacity_limit, gate_probs.shape[1]), dim=-1)
        
        # Update workspace state
        attended_squeezed = attended.squeeze(1)
        new_state = self.state_update(attended_squeezed, workspace_state)
        new_state = self.norm(new_state)
        
        return new_state, attention.squeeze(1), gate_probs


# =============================================================================
# Brain-Inspired Layer with Engram
# =============================================================================

class BrainInspiredLayer(nn.Module):
    """
    A single layer combining:
    1. Engram (conditional memory for static patterns)
    2. SNN processing (temporal computation)
    3. Attention (global context)
    4. FFN (nonlinear transformation)
    
    The flow:
    Engram → SNN → Attention → FFN
    
    Engram first: offload local patterns before deep computation
    SNN next: sparse temporal processing
    Attention: global context integration
    FFN: nonlinear transformation
    """
    
    def __init__(
        self,
        config: BrainInspiredConfig,
        layer_idx: int
    ):
        super().__init__()
        
        self.layer_idx = layer_idx
        self.hidden_dim = config.hidden_dim
        
        # Engram at specified layers
        self.use_engram = config.use_engram and (layer_idx in config.engram_layers)
        if self.use_engram:
            engram_config = EngramConfig(
                vocab_size=config.vocab_size,
                embedding_dim=config.engram_dim,
                n_gram_orders=config.engram_ngrams,
                num_heads=config.engram_heads
            )
            self.engram = EngramModule(
                config=engram_config,
                hidden_dim=config.hidden_dim
            )
        
        # SNN layer (optional)
        self.use_snn = config.use_snn
        if self.use_snn:
            self.snn = SNNLayer(
                in_features=config.hidden_dim,
                out_features=config.hidden_dim,
                num_timesteps=config.snn_timesteps,
                beta=config.snn_beta
            )
            self.snn_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        self.attn_norm = RMSNorm(config.hidden_dim)
        
        # FFN
        ffn_dim = config.hidden_dim * config.ffn_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(ffn_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
        self.ffn_norm = RMSNorm(config.hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through brain-inspired layer.
        
        Args:
            x: (batch, seq, hidden) hidden states
            token_ids: (batch, seq) for Engram lookup
            attention_mask: Optional attention mask
            
        Returns:
            output: (batch, seq, hidden) transformed states
            info: Diagnostic information
        """
        info = {}
        
        # Step 1: Engram (conditional memory)
        if self.use_engram and token_ids is not None:
            engram_out, engram_info = self.engram(token_ids, x)
            x = x + engram_out
            info['engram_gate'] = engram_info['gate_values']
        
        # Step 2: SNN processing (per position)
        if self.use_snn:
            batch, seq_len, hidden = x.shape
            
            # Process each position through SNN
            x_flat = x.view(batch * seq_len, hidden)
            snn_out, spikes = self.snn(x_flat)
            snn_out = self.snn_proj(snn_out)
            snn_out = snn_out.view(batch, seq_len, hidden)
            
            x = x + 0.1 * snn_out  # Scaled residual
            info['spike_rate'] = spikes.mean()
        
        # Step 3: Attention (global context)
        x_norm = self.attn_norm(x)
        attn_out, attn_weights = self.attention(
            x_norm, x_norm, x_norm,
            key_padding_mask=attention_mask,
            need_weights=True
        )
        x = x + attn_out
        info['attention'] = attn_weights
        
        # Step 4: FFN
        x_norm = self.ffn_norm(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        
        return x, info


# =============================================================================
# Complete Brain-Inspired Model
# =============================================================================

class BrainInspiredAI(nn.Module):
    """
    Complete Brain-Inspired AI with Engram Memory.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    BRAIN-INSPIRED AI + ENGRAM                   │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  Input → Token Embedding                                        │
    │              ↓                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │ Layer 0: Attention → FFN                                 │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │              ↓                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │ Layer 1: ENGRAM → SNN → Attention → FFN                  │  │
    │  │          ↑                                                │  │
    │  │   [O(1) N-gram lookup for static patterns]               │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │              ↓                                                  │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │ Layers 2-N: (SNN) → Attention → FFN                      │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │              ↓                                                  │
    │  Global Workspace (Working Memory Integration)                  │
    │              ↓                                                  │
    │  Output Head                                                    │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Memory Systems:
    - Engram: Semantic memory (fast lookup of static patterns)
    - Global Workspace: Working memory (capacity-limited integration)
    - SNN dynamics: Short-term memory (membrane potentials)
    """
    
    def __init__(self, config: BrainInspiredConfig):
        super().__init__()
        
        self.config = config
        
        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            self._sinusoidal_encoding(2048, config.hidden_dim),
            requires_grad=False
        )
        
        # Brain-inspired layers
        self.layers = nn.ModuleList([
            BrainInspiredLayer(config, layer_idx=i)
            for i in range(config.num_layers)
        ])
        
        # Global Workspace (working memory integration)
        if config.use_workspace:
            self.workspace = GlobalWorkspace(
                workspace_dim=config.hidden_dim,
                num_heads=config.workspace_heads,
                capacity_limit=config.capacity_limit
            )
        
        # Output layers
        self.output_norm = RMSNorm(config.hidden_dim)
        self.classifier = nn.Linear(config.hidden_dim, config.output_dim)
        
        # Optional: Language modeling head
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Track workspace state
        self.workspace_state = None
        
        self._init_weights()
    
    def _sinusoidal_encoding(self, max_len: int, dim: int) -> torch.Tensor:
        """Generate sinusoidal positional encodings."""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe = torch.zeros(1, max_len, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_intermediates: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass through brain-inspired model.
        
        Args:
            input_ids: (batch, seq) token IDs
            attention_mask: Optional attention mask
            return_intermediates: Whether to return layer info
            
        Returns:
            Dict with:
                - logits: Classification logits
                - lm_logits: Language modeling logits (optional)
                - workspace_attention: What information "won" in workspace
                - layer_info: Per-layer diagnostics (if requested)
        """
        batch, seq_len = input_ids.shape
        
        # Embed tokens
        x = self.token_embedding(input_ids)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Process through layers
        layer_infos = []
        for layer in self.layers:
            x, info = layer(x, token_ids=input_ids, attention_mask=attention_mask)
            if return_intermediates:
                layer_infos.append(info)
        
        # Global Workspace integration
        workspace_attention = None
        if self.config.use_workspace:
            # Create inputs from different "modules"
            workspace_inputs = {
                'sequential': x[:, -1, :],  # Last position (recency)
                'pooled': x.mean(dim=1),    # Average pooling (gist)
                'max': x.max(dim=1).values, # Max pooling (salience)
            }
            
            integrated, workspace_attention, capacity_scores = self.workspace(
                workspace_inputs,
                self.workspace_state
            )
            self.workspace_state = integrated.detach()
            
            # Blend workspace output with final hidden state
            final_hidden = integrated
        else:
            final_hidden = x.mean(dim=1)
        
        # Output heads
        final_hidden = self.output_norm(final_hidden)
        logits = self.classifier(final_hidden)
        
        # Language modeling (optional, for all positions)
        x_norm = self.output_norm(x)
        lm_logits = self.lm_head(x_norm)
        
        result = {
            'logits': logits,
            'lm_logits': lm_logits,
            'hidden_states': x,
            'workspace_attention': workspace_attention,
        }
        
        if return_intermediates:
            result['layer_info'] = layer_infos
        
        return result
    
    def reset_state(self):
        """Reset stateful components for new sequence."""
        self.workspace_state = None
    
    def get_engram_statistics(self) -> Dict[str, float]:
        """Get statistics about Engram usage across layers."""
        stats = {}
        for i, layer in enumerate(self.layers):
            if layer.use_engram:
                # Count parameters
                engram_params = sum(p.numel() for p in layer.engram.parameters())
                stats[f'layer_{i}_engram_params'] = engram_params
        
        total_params = sum(p.numel() for p in self.parameters())
        engram_total = sum(v for k, v in stats.items() if 'params' in k)
        stats['engram_fraction'] = engram_total / total_params
        
        return stats


# =============================================================================
# Sparsity Allocation: Finding optimal MoE/Engram balance
# =============================================================================

def compute_optimal_allocation(
    total_params: int,
    active_params: int,
    engram_ratios: List[float] = None
) -> Dict[str, Any]:
    """
    Compute sparsity allocation following the paper's U-shaped finding.
    
    The paper found optimal allocation around ρ ≈ 75-80% for MoE,
    meaning ~20-25% of sparse capacity goes to Engram.
    
    Args:
        total_params: Total parameter budget
        active_params: Activated parameters per token
        engram_ratios: Ratios to evaluate (1-ρ values)
        
    Returns:
        Allocation recommendations
    """
    if engram_ratios is None:
        engram_ratios = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
    
    sparse_budget = total_params - active_params
    
    allocations = []
    for engram_ratio in engram_ratios:
        engram_params = int(sparse_budget * engram_ratio)
        moe_params = sparse_budget - engram_params
        
        allocations.append({
            'engram_ratio': engram_ratio,
            'moe_ratio': 1 - engram_ratio,
            'engram_params': engram_params,
            'moe_params': moe_params,
            'estimated_quality': _estimate_quality(engram_ratio)
        })
    
    # Find optimal (paper suggests ~20-25% to Engram)
    optimal = max(allocations, key=lambda x: x['estimated_quality'])
    
    return {
        'allocations': allocations,
        'optimal': optimal,
        'recommendation': f"Allocate ~{optimal['engram_ratio']*100:.0f}% to Engram"
    }


def _estimate_quality(engram_ratio: float) -> float:
    """
    Estimate quality based on paper's U-shaped finding.
    
    Optimal around 0.2-0.25 engram_ratio (ρ=0.75-0.8 in paper terms).
    """
    # U-shaped curve with minimum at ~0.22
    optimal = 0.22
    quality = 1.0 - 4 * (engram_ratio - optimal) ** 2
    return max(0, quality)


# =============================================================================
# Demonstration
# =============================================================================

def demo_brain_inspired_engram():
    """Demonstrate the complete brain-inspired system with Engram."""
    
    print("=" * 70)
    print("Brain-Inspired AI with Engram Conditional Memory")
    print("=" * 70)
    
    # Configuration
    config = BrainInspiredConfig(
        vocab_size=10000,
        hidden_dim=256,
        output_dim=10,
        
        # Engram settings
        use_engram=True,
        engram_dim=128,
        engram_ngrams=[2, 3],
        engram_heads=4,
        engram_layers=[1],  # Early layer intervention
        
        # SNN settings
        use_snn=True,
        snn_timesteps=8,
        snn_beta=0.9,
        
        # Workspace settings
        use_workspace=True,
        workspace_dim=256,
        capacity_limit=7,
        
        # Architecture
        num_layers=4,
        num_heads=4
    )
    
    # Create model
    model = BrainInspiredAI(config)
    
    # Print architecture summary
    print("\n📊 Architecture Summary:")
    print(f"  Layers: {config.num_layers}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Engram at layers: {config.engram_layers}")
    print(f"  SNN timesteps: {config.snn_timesteps}")
    print(f"  Workspace capacity: {config.capacity_limit}")
    
    # Parameter breakdown
    total_params = sum(p.numel() for p in model.parameters())
    
    engram_params = 0
    for layer in model.layers:
        if layer.use_engram:
            engram_params += sum(p.numel() for p in layer.engram.parameters())
    
    snn_params = 0
    for layer in model.layers:
        if layer.use_snn:
            snn_params += sum(p.numel() for p in layer.snn.parameters())
            snn_params += sum(p.numel() for p in layer.snn_proj.parameters())
    
    workspace_params = sum(p.numel() for p in model.workspace.parameters()) if config.use_workspace else 0
    
    print(f"\n📈 Parameter Breakdown:")
    print(f"  Total: {total_params:,}")
    print(f"  Engram: {engram_params:,} ({100*engram_params/total_params:.1f}%)")
    print(f"  SNN: {snn_params:,} ({100*snn_params/total_params:.1f}%)")
    print(f"  Workspace: {workspace_params:,} ({100*workspace_params/total_params:.1f}%)")
    print(f"  Other: {total_params - engram_params - snn_params - workspace_params:,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 32
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\n🔄 Forward Pass:")
    print(f"  Input shape: {input_ids.shape}")
    
    outputs = model(input_ids, return_intermediates=True)
    
    print(f"  Classification logits: {outputs['logits'].shape}")
    print(f"  LM logits: {outputs['lm_logits'].shape}")
    print(f"  Hidden states: {outputs['hidden_states'].shape}")
    
    # Analyze Engram gating
    print(f"\n🔍 Engram Analysis:")
    for i, info in enumerate(outputs['layer_info']):
        if 'engram_gate' in info:
            gate = info['engram_gate']
            print(f"  Layer {i} gate stats:")
            print(f"    Mean: {gate.mean():.4f}")
            print(f"    Std:  {gate.std():.4f}")
            print(f"    Range: [{gate.min():.4f}, {gate.max():.4f}]")
    
    # Analyze workspace attention
    if outputs['workspace_attention'] is not None:
        print(f"\n🧠 Workspace (Working Memory) Analysis:")
        ws_attn = outputs['workspace_attention']
        print(f"  Attention to 'sequential': {ws_attn[0, 0]:.4f}")
        print(f"  Attention to 'pooled': {ws_attn[0, 1]:.4f}")
        print(f"  Attention to 'max': {ws_attn[0, 2]:.4f}")
    
    # Analyze SNN
    print(f"\n⚡ SNN Analysis:")
    for i, info in enumerate(outputs['layer_info']):
        if 'spike_rate' in info:
            print(f"  Layer {i} spike rate: {info['spike_rate']:.4f}")
    
    # Sparsity allocation analysis
    print(f"\n📐 Sparsity Allocation Analysis:")
    allocation = compute_optimal_allocation(
        total_params=total_params,
        active_params=total_params // 10  # Assume 10x sparsity
    )
    print(f"  {allocation['recommendation']}")
    print(f"  Optimal engram ratio: {allocation['optimal']['engram_ratio']:.1%}")
    
    # Layer composition
    print(f"\n🏗️ Layer Composition:")
    for i, layer in enumerate(model.layers):
        components = []
        if layer.use_engram:
            components.append("Engram")
        if layer.use_snn:
            components.append("SNN")
        components.extend(["Attention", "FFN"])
        print(f"  Layer {i}: {' → '.join(components)}")
    
    print("\n" + "=" * 70)
    print("✅ Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo_brain_inspired_engram()
