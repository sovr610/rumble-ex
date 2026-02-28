# Copyright (c) Meta Platforms, Inc. and affiliates.
# MIT License
#
# attentive_classifier_template.py
#
# AttentivePooler and AttentiveClassifier for frozen backbone probing.
# The pooler replaces global average pooling with a learned cross-attention
# mechanism that queries task-relevant spatial-temporal tokens.

from __future__ import annotations

import math
import unittest
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# AttentivePooler
# ---------------------------------------------------------------------------

class AttentivePooler(nn.Module):
    """
    Cross-attention pooling from a set of learnable query tokens into
    an encoder's patch sequence.

    Args:
        embed_dim:   Dimensionality of encoder tokens and queries.
        num_queries: Number of learnable query tokens (Q).
        num_heads:   Number of attention heads.
        depth:       Number of stacked cross-attention blocks (default=1).
        dropout:     Attention dropout probability.
    """

    def __init__(
        self,
        embed_dim: int,
        num_queries: int = 1,
        num_heads: int = 1,
        depth: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim   = embed_dim
        self.num_queries = num_queries
        self.depth       = depth

        # Learnable query tokens — shared across the batch
        self.queries = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        # Build depth cross-attention blocks
        self.blocks = nn.ModuleList([
            _CrossAttentionBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Encoder output of shape [B, N, D].

        Returns:
            Pooled queries of shape [B, Q, D].
        """
        B = x.shape[0]
        q = self.queries.expand(B, -1, -1)  # [B, Q, D]
        for block in self.blocks:
            q = block(q, x)                 # queries attend to encoder tokens
        return q                            # [B, Q, D]


class _CrossAttentionBlock(nn.Module):
    """Single cross-attention block: LayerNorm -> CrossAttn -> residual."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_q  = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn    = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_out = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, queries: Tensor, context: Tensor) -> Tensor:
        # queries: [B, Q, D],  context: [B, N, D]
        q  = self.norm_q(queries)
        kv = self.norm_kv(context)
        attended, _ = self.attn(q, kv, kv)  # [B, Q, D]
        queries = queries + attended         # residual
        queries = queries + self.ff(self.norm_out(queries))
        return queries


# ---------------------------------------------------------------------------
# AttentiveClassifier
# ---------------------------------------------------------------------------

class AttentiveClassifier(nn.Module):
    """
    Frozen-backbone classification head using AttentivePooler.

    Receives encoder patch tokens, pools them into a fixed-size representation
    via cross-attention, and projects to class logits.

    Public contract::

        classifier = AttentiveClassifier(embed_dim=1024, num_classes=174)
        logits = classifier(encoder_output)  # [B, 174]

    Args:
        embed_dim:   Dimensionality of encoder output tokens (D).
        num_classes: Number of target classes.
        num_queries: Number of query tokens in the AttentivePooler.
        depth:       Depth of the AttentivePooler.
        num_heads:   Number of attention heads.
        dropout:     Dropout on attention weights.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        num_queries: int = 1,
        depth: int = 1,
        num_heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pooler    = AttentivePooler(embed_dim, num_queries, num_heads, depth, dropout)
        self.norm      = nn.LayerNorm(embed_dim)
        self.head      = nn.Linear(embed_dim, num_classes)

        # Initialize head with small weights (standard for probing heads)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, encoder_output: Tensor) -> Tensor:
        """
        Args:
            encoder_output: [B, N, D] — patch tokens from frozen encoder.

        Returns:
            logits: [B, num_classes]
        """
        pooled = self.pooler(encoder_output)  # [B, Q, D]
        pooled = pooled.mean(dim=1)           # [B, D] — mean over queries
        pooled = self.norm(pooled)
        return self.head(pooled)              # [B, num_classes]


# ---------------------------------------------------------------------------
# Utility: build a minimal AttentiveClassifier for testing
# ---------------------------------------------------------------------------

def build_test_classifier(
    embed_dim: int = 64,
    num_classes: int = 10,
    num_queries: int = 1,
    depth: int = 1,
) -> AttentiveClassifier:
    """Return a small AttentiveClassifier suitable for unit tests."""
    return AttentiveClassifier(
        embed_dim=embed_dim,
        num_classes=num_classes,
        num_queries=num_queries,
        depth=depth,
        num_heads=1,
    )


# ---------------------------------------------------------------------------
# Self-Tests
# ---------------------------------------------------------------------------

class _TestAttentivePooler(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.embed_dim   = 64
        self.num_patches = 49   # 7x7 spatial, 1 temporal
        self.batch_size  = 4

    def test_output_shape_single_query(self):
        pooler = AttentivePooler(embed_dim=self.embed_dim, num_queries=1)
        x   = torch.randn(self.batch_size, self.num_patches, self.embed_dim)
        out = pooler(x)
        self.assertEqual(out.shape, (self.batch_size, 1, self.embed_dim))

    def test_output_shape_multi_query(self):
        pooler = AttentivePooler(embed_dim=self.embed_dim, num_queries=3)
        x   = torch.randn(self.batch_size, self.num_patches, self.embed_dim)
        out = pooler(x)
        self.assertEqual(out.shape, (self.batch_size, 3, self.embed_dim))

    def test_deep_pooler(self):
        pooler = AttentivePooler(embed_dim=self.embed_dim, num_queries=2, depth=4)
        x   = torch.randn(self.batch_size, self.num_patches, self.embed_dim)
        out = pooler(x)
        self.assertEqual(out.shape, (self.batch_size, 2, self.embed_dim))

    def test_queries_parameter_is_learnable(self):
        pooler = AttentivePooler(embed_dim=self.embed_dim, num_queries=1)
        self.assertTrue(pooler.queries.requires_grad)

    def test_output_is_not_nan(self):
        pooler = AttentivePooler(embed_dim=self.embed_dim, num_queries=1)
        x   = torch.randn(self.batch_size, self.num_patches, self.embed_dim)
        out = pooler(x)
        self.assertFalse(torch.isnan(out).any())


class _TestAttentiveClassifier(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.embed_dim   = 64
        self.num_classes = 10
        self.num_patches = 196
        self.batch_size  = 4
        self.classifier  = AttentiveClassifier(
            embed_dim=self.embed_dim,
            num_classes=self.num_classes,
        )

    def test_output_shape(self):
        x   = torch.randn(self.batch_size, self.num_patches, self.embed_dim)
        out = self.classifier(x)
        self.assertEqual(out.shape, (self.batch_size, self.num_classes))

    def test_batch_size_one(self):
        x   = torch.randn(1, self.num_patches, self.embed_dim)
        out = self.classifier(x)
        self.assertEqual(out.shape, (1, self.num_classes))

    def test_large_num_classes(self):
        clf = AttentiveClassifier(embed_dim=self.embed_dim, num_classes=1000)
        x   = torch.randn(2, self.num_patches, self.embed_dim)
        out = clf(x)
        self.assertEqual(out.shape, (2, 1000))

    def test_multi_query_produces_correct_output_shape(self):
        clf = AttentiveClassifier(
            embed_dim=self.embed_dim,
            num_classes=self.num_classes,
            num_queries=3,
        )
        x   = torch.randn(self.batch_size, self.num_patches, self.embed_dim)
        out = clf(x)
        self.assertEqual(out.shape, (self.batch_size, self.num_classes))

    def test_gradient_flows_through_head(self):
        x   = torch.randn(self.batch_size, self.num_patches, self.embed_dim)
        out = self.classifier(x)
        out.sum().backward()
        self.assertIsNotNone(self.classifier.head.weight.grad)
        self.assertFalse(
            torch.allclose(
                self.classifier.head.weight.grad,
                torch.zeros_like(self.classifier.head.weight.grad),
            )
        )

    def test_gradient_flows_through_pooler_queries(self):
        x   = torch.randn(self.batch_size, self.num_patches, self.embed_dim)
        out = self.classifier(x)
        out.sum().backward()
        self.assertIsNotNone(self.classifier.pooler.queries.grad)

    def test_no_gradient_on_frozen_encoder(self):
        """Simulate frozen encoder — encoder input should not receive grad."""
        # Create a tiny frozen encoder
        encoder = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        for p in encoder.parameters():
            p.requires_grad = False

        raw    = torch.randn(self.batch_size, self.num_patches, self.embed_dim)
        # Run encoder in no_grad — no gradients flow back
        with torch.no_grad():
            enc_out = encoder(raw)

        out = self.classifier(enc_out)
        out.sum().backward()

        for p in encoder.parameters():
            self.assertIsNone(p.grad)

    def test_output_not_nan(self):
        x   = torch.randn(self.batch_size, self.num_patches, self.embed_dim)
        out = self.classifier(x)
        self.assertFalse(torch.isnan(out).any())

    def test_deterministic_with_seed(self):
        torch.manual_seed(7)
        x    = torch.randn(2, self.num_patches, self.embed_dim)
        out1 = self.classifier(x)
        torch.manual_seed(7)
        x    = torch.randn(2, self.num_patches, self.embed_dim)
        out2 = self.classifier(x)
        torch.testing.assert_close(out1, out2)

    def test_vit_giant_scale(self):
        """Verify the classifier handles ViT-Giant embed_dim=1408."""
        clf = AttentiveClassifier(embed_dim=1408, num_classes=174, num_heads=4)
        x   = torch.randn(2, 1568, 1408)
        out = clf(x)
        self.assertEqual(out.shape, (2, 174))


if __name__ == "__main__":
    print("Running AttentiveClassifier self-tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(_TestAttentivePooler)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(_TestAttentiveClassifier))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\nAll self-tests passed.")
    else:
        raise SystemExit(1)
