"""
Engram: Conditional Memory via Scalable Lookup
==============================================

Implementation based on DeepSeek's paper "Conditional Memory via Scalable Lookup:
A New Axis of Sparsity for Large Language Models"

Key concepts:
- N-gram embeddings with O(1) hash-based lookup
- Tokenizer compression for semantic density
- Multi-head hashing to reduce collisions
- Context-aware gating for disambiguation
- Deterministic addressing enables prefetching/offloading

This module complements conditional computation (MoE/SNN) with conditional memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import hashlib


@dataclass
class EngramConfig:
    """Configuration for Engram module."""
    vocab_size: int = 50257          # Original vocabulary size
    compressed_vocab_size: int = None # After tokenizer compression (~77% of original)
    embedding_dim: int = 256          # Dimension per N-gram embedding
    n_gram_orders: List[int] = None   # Which N-grams to use (default: [2, 3])
    num_heads: int = 8                # Number of hash heads per N-gram order
    table_size: int = 1000003         # Prime number for hash table size
    conv_kernel_size: int = 4         # Depthwise conv kernel
    conv_dilation: int = 3            # Dilation = max N-gram order
    use_tokenizer_compression: bool = True
    
    def __post_init__(self):
        if self.n_gram_orders is None:
            self.n_gram_orders = [2, 3]
        if self.compressed_vocab_size is None:
            # ~23% reduction from tokenizer compression
            self.compressed_vocab_size = int(self.vocab_size * 0.77)


class TokenizerCompression:
    """
    Vocabulary projection layer that collapses semantically equivalent tokens.
    
    Maps raw token IDs to canonical identifiers based on:
    - NFKC normalization
    - Lowercasing
    - Whitespace normalization
    
    This achieves ~23% vocabulary reduction while preserving semantic content.
    """
    
    def __init__(self, vocab_size: int, compressed_size: int):
        self.vocab_size = vocab_size
        self.compressed_size = compressed_size
        
        # Create projection mapping (in practice, this would be computed
        # from the actual tokenizer vocabulary)
        # For now, we use a hash-based simulation
        self._build_projection_table()
    
    def _build_projection_table(self):
        """Build the surjective mapping P: V -> V'."""
        # In production, this would analyze actual token text
        # Here we simulate with deterministic hashing
        self.projection = torch.zeros(self.vocab_size, dtype=torch.long)
        
        for token_id in range(self.vocab_size):
            # Simulate normalization by mapping to compressed space
            # Real implementation would use actual token strings
            canonical_id = token_id % self.compressed_size
            self.projection[token_id] = canonical_id
    
    def compress(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Map raw token IDs to canonical IDs.
        
        Args:
            token_ids: (batch, seq_len) raw token IDs
            
        Returns:
            Compressed token IDs in range [0, compressed_size)
        """
        device = token_ids.device
        return self.projection.to(device)[token_ids]


class MultiHeadHash:
    """
    Multi-head hashing for N-gram to embedding index mapping.
    
    Uses K distinct hash functions per N-gram order to reduce collisions.
    Each hash head maps to a separate embedding table.
    
    Hash function: multiplicative-XOR hash
    φ(g) = (Σ_i c_i * x_i) XOR seed mod M
    
    where c_i are position-dependent coefficients.
    """
    
    def __init__(
        self,
        n_gram_order: int,
        num_heads: int,
        table_size: int,
        seed: int = 42
    ):
        self.n = n_gram_order
        self.num_heads = num_heads
        self.table_size = table_size  # Should be prime
        
        # Generate random coefficients for each head
        rng = np.random.RandomState(seed)
        
        # Coefficients for multiplicative hash: shape (num_heads, n_gram_order)
        self.coefficients = torch.tensor(
            rng.randint(1, table_size, size=(num_heads, n_gram_order)),
            dtype=torch.long
        )
        
        # XOR seeds for each head
        self.seeds = torch.tensor(
            rng.randint(0, table_size, size=(num_heads,)),
            dtype=torch.long
        )
    
    def hash(self, n_grams: torch.Tensor) -> torch.Tensor:
        """
        Hash N-grams to embedding indices.
        
        Args:
            n_grams: (batch, seq_len, n) tensor of token IDs forming N-grams
            
        Returns:
            (batch, seq_len, num_heads) tensor of embedding indices
        """
        device = n_grams.device
        batch, seq_len, n = n_grams.shape
        
        coeffs = self.coefficients.to(device)  # (num_heads, n)
        seeds = self.seeds.to(device)          # (num_heads,)
        
        # Compute weighted sum: (batch, seq_len, num_heads)
        # n_grams: (batch, seq_len, n) -> (batch, seq_len, 1, n)
        # coeffs: (num_heads, n) -> (1, 1, num_heads, n)
        weighted = (n_grams.unsqueeze(2) * coeffs.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        
        # XOR with seeds and take modulo
        hashed = (weighted ^ seeds.unsqueeze(0).unsqueeze(0)) % self.table_size
        
        return hashed


class EngramEmbedding(nn.Module):
    """
    N-gram embedding tables with multi-head hashing.
    
    For each N-gram order n, maintains K embedding tables of size M.
    Retrieved embeddings are concatenated across all orders and heads.
    """
    
    def __init__(self, config: EngramConfig):
        super().__init__()
        
        self.config = config
        self.n_gram_orders = config.n_gram_orders
        self.num_heads = config.num_heads
        
        # Dimension per individual embedding
        total_retrievals = len(self.n_gram_orders) * config.num_heads
        self.dim_per_embedding = config.embedding_dim // total_retrievals
        
        # Create hash functions for each N-gram order
        self.hashers = {
            n: MultiHeadHash(
                n_gram_order=n,
                num_heads=config.num_heads,
                table_size=config.table_size,
                seed=42 + n
            )
            for n in self.n_gram_orders
        }
        
        # Create embedding tables: one per (n_gram_order, head)
        self.embeddings = nn.ModuleDict()
        for n in self.n_gram_orders:
            for k in range(config.num_heads):
                key = f"ngram{n}_head{k}"
                self.embeddings[key] = nn.Embedding(
                    num_embeddings=config.table_size,
                    embedding_dim=self.dim_per_embedding
                )
        
        # Tokenizer compression
        if config.use_tokenizer_compression:
            self.compressor = TokenizerCompression(
                config.vocab_size,
                config.compressed_vocab_size
            )
        else:
            self.compressor = None
    
    def extract_ngrams(
        self,
        token_ids: torch.Tensor,
        n: int
    ) -> torch.Tensor:
        """
        Extract suffix N-grams from token sequence.
        
        Args:
            token_ids: (batch, seq_len) compressed token IDs
            n: N-gram order
            
        Returns:
            (batch, seq_len, n) tensor where [b, t, :] = (x_{t-n+1}, ..., x_t)
        """
        batch, seq_len = token_ids.shape
        device = token_ids.device
        
        # Pad with zeros for positions < n
        padded = F.pad(token_ids, (n - 1, 0), value=0)  # (batch, seq_len + n - 1)
        
        # Extract sliding windows
        n_grams = torch.stack([
            padded[:, i:i + seq_len]
            for i in range(n)
        ], dim=-1)  # (batch, seq_len, n)
        
        return n_grams
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Retrieve N-gram embeddings for input sequence.
        
        Args:
            token_ids: (batch, seq_len) raw token IDs
            
        Returns:
            (batch, seq_len, embedding_dim) concatenated embeddings
        """
        # Apply tokenizer compression
        if self.compressor is not None:
            compressed_ids = self.compressor.compress(token_ids)
        else:
            compressed_ids = token_ids
        
        all_embeddings = []
        
        for n in self.n_gram_orders:
            # Extract N-grams: (batch, seq_len, n)
            n_grams = self.extract_ngrams(compressed_ids, n)
            
            # Hash to indices: (batch, seq_len, num_heads)
            indices = self.hashers[n].hash(n_grams)
            
            # Retrieve from each head's table
            for k in range(self.num_heads):
                key = f"ngram{n}_head{k}"
                head_indices = indices[:, :, k]  # (batch, seq_len)
                head_emb = self.embeddings[key](head_indices)  # (batch, seq_len, dim_per_emb)
                all_embeddings.append(head_emb)
        
        # Concatenate all retrieved embeddings
        return torch.cat(all_embeddings, dim=-1)  # (batch, seq_len, embedding_dim)


class ContextAwareGating(nn.Module):
    """
    Context-aware gating mechanism for Engram.
    
    Uses the hidden state (which has global context via attention) as Query
    and retrieved memory as Key/Value. This enables:
    - Disambiguation of polysemous patterns
    - Suppression of hash collision noise
    - Dynamic relevance weighting
    
    Gate α = σ(RMSNorm(h)ᵀ RMSNorm(k) / √d)
    Output = α · v
    """
    
    def __init__(
        self,
        hidden_dim: int,
        memory_dim: int,
        num_branches: int = 1  # For multi-branch architectures like mHC
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.num_branches = num_branches
        
        # Shared Value projection
        self.W_V = nn.Linear(memory_dim, hidden_dim, bias=False)
        
        # Branch-specific Key projections (enables different gating per branch)
        self.W_K = nn.ModuleList([
            nn.Linear(memory_dim, hidden_dim, bias=False)
            for _ in range(num_branches)
        ])
        
        # RMSNorm for stability
        self.query_norm = RMSNorm(hidden_dim)
        self.key_norm = RMSNorm(hidden_dim)
        
        self.scale = hidden_dim ** -0.5
    
    def forward(
        self,
        hidden_states: torch.Tensor,  # (batch, seq, hidden) or list for multi-branch
        memory: torch.Tensor,          # (batch, seq, memory_dim)
        branch_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply context-aware gating.
        
        Args:
            hidden_states: Current hidden state with global context
            memory: Retrieved N-gram embeddings
            branch_idx: Which branch (for multi-branch architectures)
            
        Returns:
            gated_output: α · v
            gate_values: The gating scalars α (for visualization)
        """
        # Project memory to key and value
        k = self.W_K[branch_idx](memory)  # (batch, seq, hidden)
        v = self.W_V(memory)              # (batch, seq, hidden)
        
        # Normalize query and key
        q_norm = self.query_norm(hidden_states)  # (batch, seq, hidden)
        k_norm = self.key_norm(k)
        
        # Compute gating scalar via dot product
        # α = σ((q · k) / √d)
        gate_logits = (q_norm * k_norm).sum(dim=-1, keepdim=True) * self.scale
        alpha = torch.sigmoid(gate_logits)  # (batch, seq, 1)
        
        # Apply gate
        gated_output = alpha * v  # (batch, seq, hidden)
        
        return gated_output, alpha.squeeze(-1)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class EngramModule(nn.Module):
    """
    Complete Engram conditional memory module.
    
    Architecture:
    1. N-gram extraction with tokenizer compression
    2. Multi-head hash lookup from embedding tables
    3. Context-aware gating using hidden state
    4. Depthwise causal convolution for receptive field expansion
    5. Residual connection to backbone
    
    This module should be placed at early layers (e.g., layer 2)
    to offload local pattern reconstruction before deep computation.
    """
    
    def __init__(
        self,
        config: EngramConfig,
        hidden_dim: int,
        num_branches: int = 1
    ):
        super().__init__()
        
        self.config = config
        self.hidden_dim = hidden_dim
        self.num_branches = num_branches
        
        # N-gram embedding retrieval
        self.ngram_embedding = EngramEmbedding(config)
        
        # Context-aware gating
        self.gating = ContextAwareGating(
            hidden_dim=hidden_dim,
            memory_dim=config.embedding_dim,
            num_branches=num_branches
        )
        
        # Depthwise causal convolution for receptive field expansion
        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=config.conv_kernel_size,
            padding=(config.conv_kernel_size - 1) * config.conv_dilation,  # Causal padding
            dilation=config.conv_dilation,
            groups=hidden_dim  # Depthwise
        )
        
        # Pre-conv normalization
        self.conv_norm = RMSNorm(hidden_dim)
        
        # Initialize conv to identity (zero init for smooth training start)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
    
    def forward(
        self,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        branch_idx: int = 0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply Engram conditional memory.
        
        Args:
            token_ids: (batch, seq_len) input token IDs
            hidden_states: (batch, seq_len, hidden_dim) current hidden states
            branch_idx: Which branch for multi-branch architectures
            
        Returns:
            output: (batch, seq_len, hidden_dim) to be added to residual stream
            info: Dictionary with gating values and other diagnostics
        """
        batch, seq_len = token_ids.shape
        
        # Step 1: Retrieve N-gram embeddings via hash lookup
        memory = self.ngram_embedding(token_ids)  # (batch, seq, memory_dim)
        
        # Step 2: Context-aware gating
        gated, gate_values = self.gating(
            hidden_states, memory, branch_idx
        )  # (batch, seq, hidden)
        
        # Step 3: Depthwise causal convolution
        # Normalize before conv
        gated_norm = self.conv_norm(gated)
        
        # Conv expects (batch, channels, seq)
        conv_input = gated_norm.transpose(1, 2)
        conv_output = self.conv(conv_input)
        
        # Truncate to maintain causality (remove future padding)
        conv_output = conv_output[:, :, :seq_len]
        conv_output = conv_output.transpose(1, 2)  # Back to (batch, seq, hidden)
        
        # SiLU activation + residual
        output = F.silu(conv_output) + gated
        
        info = {
            'gate_values': gate_values,
            'memory_norm': memory.norm(dim=-1).mean(),
        }
        
        return output, info


class EngramLayer(nn.Module):
    """
    A transformer-style layer augmented with Engram.
    
    Structure: Engram -> Attention -> MoE/FFN
    
    Engram is applied BEFORE attention to offload local pattern
    reconstruction, freeing attention to focus on global dependencies.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        engram_config: EngramConfig,
        num_heads: int = 8,
        ffn_dim: int = None,
        dropout: float = 0.1,
        use_engram: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_engram = use_engram
        
        if ffn_dim is None:
            ffn_dim = 4 * hidden_dim
        
        # Engram conditional memory
        if use_engram:
            self.engram = EngramModule(
                config=engram_config,
                hidden_dim=hidden_dim
            )
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = RMSNorm(hidden_dim)
        
        # FFN (could be replaced with MoE)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = RMSNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through Engram-augmented layer.
        
        Args:
            x: (batch, seq, hidden) input hidden states
            token_ids: (batch, seq) input token IDs for Engram lookup
            attention_mask: Optional attention mask
            
        Returns:
            output: (batch, seq, hidden) output hidden states
            layer_info: Diagnostic information
        """
        info = {}
        
        # Step 1: Engram (conditional memory)
        if self.use_engram:
            engram_output, engram_info = self.engram(token_ids, x)
            x = x + engram_output  # Residual
            info['engram'] = engram_info
        
        # Step 2: Self-attention
        x_norm = self.attn_norm(x)
        attn_output, attn_weights = self.attention(
            x_norm, x_norm, x_norm,
            key_padding_mask=attention_mask,
            need_weights=True
        )
        x = x + attn_output
        info['attention_weights'] = attn_weights
        
        # Step 3: FFN
        x_norm = self.ffn_norm(x)
        ffn_output = self.ffn(x_norm)
        x = x + ffn_output
        
        return x, info


# =============================================================================
# Example: Complete model with Engram at strategic layers
# =============================================================================

class EngramTransformer(nn.Module):
    """
    Transformer model with Engram at early layers.
    
    The paper found optimal Engram placement at layers 2 and 15 (for 30-layer model).
    Early placement offloads local patterns; deeper placement provides refinement.
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        engram_layers: List[int] = None,  # Which layers get Engram
        engram_config: EngramConfig = None
    ):
        super().__init__()
        
        if engram_layers is None:
            # Default: layer 2 (early intervention)
            engram_layers = [1]  # 0-indexed
        
        if engram_config is None:
            engram_config = EngramConfig(vocab_size=vocab_size)
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, 2048, hidden_dim)  # Max sequence length
        )
        
        # Transformer layers (with Engram at specified layers)
        self.layers = nn.ModuleList([
            EngramLayer(
                hidden_dim=hidden_dim,
                engram_config=engram_config,
                num_heads=num_heads,
                use_engram=(i in engram_layers)
            )
            for i in range(num_layers)
        ])
        
        # Output head
        self.output_norm = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embedding.weight
        
        self.engram_layers = engram_layers
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: (batch, seq) token IDs
            attention_mask: Optional padding mask
            
        Returns:
            Dict with logits and layer information
        """
        batch, seq_len = input_ids.shape
        
        # Embed tokens
        x = self.token_embedding(input_ids)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Process through layers
        layer_infos = []
        for layer in self.layers:
            x, info = layer(x, input_ids, attention_mask)
            layer_infos.append(info)
        
        # Output
        x = self.output_norm(x)
        logits = self.lm_head(x)
        
        return {
            'logits': logits,
            'layer_infos': layer_infos
        }


# =============================================================================
# Demonstration
# =============================================================================

def demo_engram():
    """Demonstrate Engram module functionality."""
    
    print("=" * 60)
    print("Engram Conditional Memory Demo")
    print("=" * 60)
    
    # Configuration
    config = EngramConfig(
        vocab_size=10000,
        embedding_dim=128,
        n_gram_orders=[2, 3],
        num_heads=4,
        table_size=100003  # Prime
    )
    
    hidden_dim = 256
    batch_size = 4
    seq_len = 32
    
    # Create module
    engram = EngramModule(config, hidden_dim)
    
    # Sample inputs
    token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Forward pass
    output, info = engram(token_ids, hidden_states)
    
    print(f"\nInput shapes:")
    print(f"  token_ids: {token_ids.shape}")
    print(f"  hidden_states: {hidden_states.shape}")
    
    print(f"\nOutput shapes:")
    print(f"  output: {output.shape}")
    
    print(f"\nGating statistics:")
    gate_vals = info['gate_values']
    print(f"  Gate values shape: {gate_vals.shape}")
    print(f"  Mean gate: {gate_vals.mean():.4f}")
    print(f"  Std gate: {gate_vals.std():.4f}")
    print(f"  Min gate: {gate_vals.min():.4f}")
    print(f"  Max gate: {gate_vals.max():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in engram.parameters())
    embedding_params = sum(
        p.numel() for name, p in engram.named_parameters()
        if 'embedding' in name
    )
    
    print(f"\nParameter counts:")
    print(f"  Total: {total_params:,}")
    print(f"  Embeddings: {embedding_params:,} ({100*embedding_params/total_params:.1f}%)")
    print(f"  Other: {total_params - embedding_params:,}")
    
    # Test full transformer with Engram
    print("\n" + "=" * 60)
    print("Full Transformer with Engram")
    print("=" * 60)
    
    model = EngramTransformer(
        vocab_size=10000,
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        engram_layers=[1, 4]  # Engram at layers 2 and 5 (0-indexed)
    )
    
    outputs = model(token_ids)
    
    print(f"\nModel configuration:")
    print(f"  Layers: 6")
    print(f"  Engram at layers: [1, 4]")
    print(f"  Hidden dim: 256")
    
    print(f"\nOutput logits shape: {outputs['logits'].shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    
    # Show which layers have Engram
    print(f"\nLayer Engram status:")
    for i, layer in enumerate(model.layers):
        has_engram = "✓ Engram" if layer.use_engram else "  Standard"
        print(f"  Layer {i}: {has_engram}")


if __name__ == "__main__":
    demo_engram()
