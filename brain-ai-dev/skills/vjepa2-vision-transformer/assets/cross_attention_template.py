"""
Cross-Attention and Attentive Pooler for V-JEPA 2 Vision Transformer.

Includes:
- CrossAttention: Separate Q (from learnable queries) and KV (from encoder output)
- AttentivePooler: Learns a fixed pool of query tokens that attend to encoder features,
  optionally followed by a classification head (nn.Linear).

The AttentivePooler is the standard probing head for V-JEPA 2 downstream evaluation:
    pooler = AttentivePooler(num_queries=1, embed_dim=1024, num_heads=16)
    logits = pooler(encoder_output)  # [B, num_classes]

Self-tests run when executed directly: python cross_attention_template.py
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CrossAttention
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """
    Cross-attention module where Q comes from one source and KV from another.

    Standard use in V-JEPA 2:
        - queries: learnable tokens (from AttentivePooler)
        - encoder: frozen or fine-tuned ViT encoder output

    The Q and KV projections are separate linear layers (not a single fused QKV),
    because Q and KV have different sequence lengths.

    Args:
        dim: Embedding dimension (shared between queries and encoder output).
        num_heads: Number of attention heads.
        qkv_bias: Whether to include bias in Q, K, V projections.
        use_sdpa: Use F.scaled_dot_product_attention if available (PyTorch 2.0+).
        attn_drop: Dropout on attention weights.
        proj_drop: Dropout on the output projection.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_sdpa: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, (
            f"dim={dim} must be divisible by num_heads={num_heads}"
        )
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.use_sdpa  = use_sdpa

        # Separate Q projection (for query tokens) and KV projection (for encoder tokens)
        self.q  = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)

        self.proj      = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in [self.q, self.kv, self.proj]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        queries: torch.Tensor,  # [B, Nq, D]  learnable query tokens
        encoder: torch.Tensor,  # [B, Nkv, D] ViT encoder output
    ) -> torch.Tensor:
        """
        Args:
            queries: Query tokens, shape [B, Nq, D].
            encoder: Key-value source, shape [B, Nkv, D].

        Returns:
            Output tensor, shape [B, Nq, D].
        """
        B, Nq, D = queries.shape
        Nkv = encoder.shape[1]

        # Project queries to multi-head format: [B, H, Nq, head_dim]
        q = self.q(queries)
        q = q.reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Project encoder tokens to keys and values: [B, H, Nkv, head_dim] each
        kv = self.kv(encoder)                                        # [B, Nkv, 2*D]
        kv = kv.reshape(B, Nkv, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)                              # [2, B, H, Nkv, head_dim]
        k, v = kv.unbind(0)

        if self.use_sdpa:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                scale=self.scale,
            )  # [B, H, Nq, head_dim]
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, Nq, Nkv]
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v                                  # [B, H, Nq, head_dim]

        # Merge heads: [B, H, Nq, head_dim] -> [B, Nq, D]
        out = out.transpose(1, 2).reshape(B, Nq, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# ---------------------------------------------------------------------------
# AttentivePooler
# ---------------------------------------------------------------------------

class AttentivePooler(nn.Module):
    """
    Attentive pooler for downstream task probing.

    Learns a fixed set of query tokens that attend over the ViT encoder output.
    This replaces global average pooling with learned, content-adaptive pooling.

    Architecture:
        queries (learnable) [B, num_queries, D]
              |
              v
        LayerNorm -> CrossAttention <- encoder_output [B, N, D]
              |
              v
        [B, num_queries, D]
              |
              v (optional)
        nn.Linear -> [B, num_classes]  (if num_classes > 0)

    Standard V-JEPA 2 probing uses num_queries=1 and a linear probe on top.

    Args:
        num_queries: Number of learnable query tokens (typically 1).
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        num_classes: If > 0, append a linear classification head.
                     Output will be [B, num_classes] instead of [B, num_queries, D].
        mlp_ratio: If > 0, add a post-attention MLP (for richer pooling).
        qkv_bias: Bias in attention projections.
        use_sdpa: Use SDPA backend.
        init_std: Standard deviation for query token initialization.
    """

    def __init__(
        self,
        num_queries: int = 1,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_classes: int = 0,
        mlp_ratio: float = 0.0,
        qkv_bias: bool = True,
        use_sdpa: bool = True,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim   = embed_dim
        self.num_classes = num_classes

        # Learnable query tokens: [1, num_queries, D] (broadcast over batch)
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_queries, embed_dim)
        )

        # Pre-norm for the query tokens before cross-attention
        self.query_norm = nn.LayerNorm(embed_dim)
        # Pre-norm for the encoder features (keys/values)
        self.feat_norm = nn.LayerNorm(embed_dim)

        # Cross-attention: queries attend to encoder features
        self.cross_attn = CrossAttention(
            dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_sdpa=use_sdpa,
        )

        # Optional post-attention MLP for richer representation
        if mlp_ratio > 0.0:
            hidden_dim = int(embed_dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim),
            )
        else:
            self.mlp = None

        # Optional classification head (linear probe)
        if num_classes > 0:
            self.head = nn.Linear(embed_dim, num_classes)
        else:
            self.head = None

        self._init_weights(init_std)

    def _init_weights(self, std: float) -> None:
        # Initialize query tokens with truncated normal
        nn.init.trunc_normal_(self.query_tokens, std=std)

        # Initialize cross-attention weights
        self.cross_attn._init_weights()

        # Initialize MLP if present
        if self.mlp is not None:
            for module in self.mlp.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

        # Initialize classification head
        if self.head is not None:
            nn.init.zeros_(self.head.weight)
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x: torch.Tensor,                         # [B, N, D] encoder output
        query_mask: Optional[torch.Tensor] = None,  # reserved for future use
    ) -> torch.Tensor:
        """
        Args:
            x: ViT encoder output, shape [B, N, D].
            query_mask: Not currently used; reserved for masked query support.

        Returns:
            If num_classes > 0: logits of shape [B, num_classes].
            Otherwise: pooled features of shape [B, num_queries, D].
        """
        B = x.shape[0]

        # Expand learnable queries to batch size: [B, num_queries, D]
        queries = self.query_tokens.expand(B, -1, -1)

        # Pre-normalize queries and encoder features
        q = self.query_norm(queries)
        kv = self.feat_norm(x)

        # Cross-attention: queries attend over all encoder tokens
        out = self.cross_attn(q, kv)  # [B, num_queries, D]

        # Residual connection: combine with original queries
        out = out + queries

        # Optional MLP
        if self.mlp is not None:
            out = out + self.mlp(out)

        # Classification head: squeeze query dim when num_queries=1
        if self.head is not None:
            if self.num_queries == 1:
                out = out.squeeze(1)  # [B, D]
            else:
                out = out.mean(dim=1)  # [B, D]  -- average multiple queries
            out = self.head(out)      # [B, num_classes]

        return out

    def extra_repr(self) -> str:
        return (
            f"num_queries={self.num_queries}, embed_dim={self.embed_dim}, "
            f"num_classes={self.num_classes}"
        )


# ---------------------------------------------------------------------------
# Self-Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Cross-Attention Template -- Self-Tests")
    print("=" * 60)

    failures = []

    # ------------------------------------------------------------------
    # Test 1: CrossAttention output shape
    # ------------------------------------------------------------------
    print("\n[1] CrossAttention output shape [B, Nq, D]...")
    try:
        for B, Nq, Nkv, D, H in [
            (2, 8,   196, 768,  12),
            (1, 1,   784, 1024, 16),
            (4, 16,  49,  384,  6),
        ]:
            ca = CrossAttention(dim=D, num_heads=H)
            queries = torch.randn(B, Nq, D)
            encoder = torch.randn(B, Nkv, D)
            out = ca(queries, encoder)
            assert out.shape == (B, Nq, D), (
                f"Expected ({B},{Nq},{D}), got {out.shape}"
            )
        print("   PASS: all configurations produce correct output shape")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("CrossAttention shape")

    # ------------------------------------------------------------------
    # Test 2: CrossAttention SDPA matches manual
    # ------------------------------------------------------------------
    print("\n[2] CrossAttention SDPA matches manual implementation...")
    try:
        torch.manual_seed(0)
        B, Nq, Nkv, D, H = 2, 4, 32, 256, 8
        ca_manual = CrossAttention(dim=D, num_heads=H, use_sdpa=False)
        ca_sdpa   = CrossAttention(dim=D, num_heads=H, use_sdpa=True)
        ca_sdpa.load_state_dict(ca_manual.state_dict())

        # Set to inference mode for deterministic comparison
        ca_manual.training = False
        ca_sdpa.training   = False

        queries = torch.randn(B, Nq, D)
        encoder = torch.randn(B, Nkv, D)
        with torch.no_grad():
            out_manual = ca_manual(queries, encoder)
            out_sdpa   = ca_sdpa(queries, encoder)

        max_diff = (out_manual - out_sdpa).abs().max().item()
        assert max_diff < 1e-4, f"Max diff: {max_diff:.2e}"
        print(f"   PASS: max diff = {max_diff:.2e}")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("CrossAttention SDPA vs manual")

    # ------------------------------------------------------------------
    # Test 3: CrossAttention Q and KV from different sources
    # ------------------------------------------------------------------
    print("\n[3] CrossAttention Q and KV independence...")
    try:
        B, Nq, Nkv, D, H = 2, 4, 16, 256, 8
        ca = CrossAttention(dim=D, num_heads=H)
        ca.training = False

        queries = torch.randn(B, Nq, D)
        encoder1 = torch.randn(B, Nkv, D)
        encoder2 = torch.randn(B, Nkv, D)

        with torch.no_grad():
            out1 = ca(queries, encoder1)
            out2 = ca(queries, encoder2)

        # Same queries, different encoders -> different outputs
        assert not torch.allclose(out1, out2), \
            "Different encoder outputs should produce different cross-attn outputs"
        # Both have same shape
        assert out1.shape == out2.shape == (B, Nq, D)
        print("   PASS: different encoder inputs produce different outputs")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("CrossAttention independence")

    # ------------------------------------------------------------------
    # Test 4: AttentivePooler single query output shape
    # ------------------------------------------------------------------
    print("\n[4] AttentivePooler single query output shape...")
    try:
        for B, N, D, H in [(2, 196, 768, 12), (1, 784, 1024, 16)]:
            pooler = AttentivePooler(num_queries=1, embed_dim=D, num_heads=H)
            pooler.training = False
            x = torch.randn(B, N, D)
            with torch.no_grad():
                out = pooler(x)
            # No classification head: output is [B, 1, D]
            assert out.shape == (B, 1, D), (
                f"Expected ({B}, 1, {D}), got {out.shape}"
            )
        print("   PASS")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("AttentivePooler single query shape")

    # ------------------------------------------------------------------
    # Test 5: AttentivePooler multiple queries output shape
    # ------------------------------------------------------------------
    print("\n[5] AttentivePooler multiple queries output shape...")
    try:
        B, N, D, H, Nq = 2, 196, 768, 12, 8
        pooler = AttentivePooler(num_queries=Nq, embed_dim=D, num_heads=H)
        pooler.training = False
        x = torch.randn(B, N, D)
        with torch.no_grad():
            out = pooler(x)
        assert out.shape == (B, Nq, D), f"Expected ({B},{Nq},{D}), got {out.shape}"
        print(f"   PASS: output shape {out.shape}")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("AttentivePooler multi-query shape")

    # ------------------------------------------------------------------
    # Test 6: AttentivePooler with classification head
    # ------------------------------------------------------------------
    print("\n[6] AttentivePooler with classification head...")
    try:
        B, N, D, H, num_classes = 2, 196, 768, 12, 400
        pooler = AttentivePooler(
            num_queries=1, embed_dim=D, num_heads=H, num_classes=num_classes
        )
        pooler.training = False
        x = torch.randn(B, N, D)
        with torch.no_grad():
            logits = pooler(x)
        assert logits.shape == (B, num_classes), (
            f"Expected ({B},{num_classes}), got {logits.shape}"
        )
        print(f"   PASS: logits shape {logits.shape}")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("AttentivePooler classifier shape")

    # ------------------------------------------------------------------
    # Test 7: AttentivePooler with multi-query classification head
    # ------------------------------------------------------------------
    print("\n[7] AttentivePooler multi-query classification (mean pooling)...")
    try:
        B, N, D, H, Nq, num_classes = 2, 196, 768, 12, 4, 1000
        pooler = AttentivePooler(
            num_queries=Nq, embed_dim=D, num_heads=H, num_classes=num_classes
        )
        pooler.training = False
        x = torch.randn(B, N, D)
        with torch.no_grad():
            logits = pooler(x)
        assert logits.shape == (B, num_classes), (
            f"Expected ({B},{num_classes}), got {logits.shape}"
        )
        print(f"   PASS: logits shape {logits.shape} (mean over {Nq} queries)")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("AttentivePooler multi-query classifier")

    # ------------------------------------------------------------------
    # Test 8: AttentivePooler query tokens are learnable
    # ------------------------------------------------------------------
    print("\n[8] AttentivePooler query tokens are learnable parameters...")
    try:
        pooler = AttentivePooler(num_queries=1, embed_dim=768, num_heads=12)
        assert pooler.query_tokens.requires_grad, \
            "query_tokens must require gradients (they are learned)"
        assert pooler.query_tokens.shape == (1, 1, 768), \
            f"Wrong query_tokens shape: {pooler.query_tokens.shape}"
        print(f"   PASS: query_tokens shape {pooler.query_tokens.shape}, requires_grad=True")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("AttentivePooler learnable queries")

    # ------------------------------------------------------------------
    # Test 9: AttentivePooler with optional MLP
    # ------------------------------------------------------------------
    print("\n[9] AttentivePooler with optional MLP head...")
    try:
        B, N, D, H = 2, 196, 768, 12
        pooler = AttentivePooler(
            num_queries=1, embed_dim=D, num_heads=H, mlp_ratio=4.0
        )
        pooler.training = False
        assert pooler.mlp is not None, "mlp should be initialized when mlp_ratio > 0"
        x = torch.randn(B, N, D)
        with torch.no_grad():
            out = pooler(x)
        assert out.shape == (B, 1, D)
        print(f"   PASS: output shape {out.shape}")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("AttentivePooler with MLP")

    # ------------------------------------------------------------------
    # Test 10: AttentivePooler is differentiable (gradient flows)
    # ------------------------------------------------------------------
    print("\n[10] AttentivePooler gradient flow...")
    try:
        B, N, D = 2, 196, 768
        pooler = AttentivePooler(num_queries=1, embed_dim=D, num_heads=12, num_classes=100)
        x = torch.randn(B, N, D, requires_grad=True)
        logits = pooler(x)
        loss = logits.sum()
        loss.backward()

        assert x.grad is not None, "Gradient should flow back to encoder output"
        assert not x.grad.isnan().any(), "Gradient contains NaN"
        print("   PASS: gradients flow correctly")
    except AssertionError as e:
        print(f"   FAIL: {e}")
        failures.append("AttentivePooler gradient")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {len(failures)} test(s): {failures}")
        sys.exit(1)
    else:
        print("ALL 10 TESTS PASSED")
        sys.exit(0)
