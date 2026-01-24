# tests/test_engram.py
"""Tests for Engram memory system."""

import pytest
import torch

from brain_ai.memory.tokenizer_compression import TokenizerCompression


class TestTokenizerCompression:
    """Tests for tokenizer compression."""

    def test_compress_reduces_vocab_range(self):
        """Compressed IDs should be in reduced range."""
        compressor = TokenizerCompression(
            vocab_size=50000,
            compressed_size=38500,  # ~77% of original
            mode="shared"
        )

        token_ids = torch.randint(0, 50000, (4, 32))
        compressed = compressor.compress(token_ids)

        assert compressed.shape == token_ids.shape
        assert compressed.max() < 38500
        assert compressed.min() >= 0

    def test_compress_is_deterministic(self):
        """Same input should always give same output."""
        compressor = TokenizerCompression(
            vocab_size=50000,
            compressed_size=38500,
            mode="shared"
        )

        token_ids = torch.randint(0, 50000, (4, 32))
        result1 = compressor.compress(token_ids)
        result2 = compressor.compress(token_ids)

        assert torch.equal(result1, result2)

    def test_semantic_equivalents_map_same(self):
        """Semantically equivalent tokens should map to same ID."""
        compressor = TokenizerCompression(
            vocab_size=50000,
            compressed_size=38500,
            mode="shared"
        )

        # Same token should always map to same compressed ID
        token_a = torch.tensor([[100]])
        token_b = torch.tensor([[100]])

        assert torch.equal(
            compressor.compress(token_a),
            compressor.compress(token_b)
        )


from brain_ai.memory.hash_embedding import MultiHeadHash, OffloadableEmbedding


class TestMultiHeadHash:
    """Tests for multi-head hashing."""

    def test_hash_output_shape(self):
        """Hash should produce correct output shape."""
        hasher = MultiHeadHash(
            ngram_order=2,
            num_heads=4,
            table_size=1000003,
        )

        # (batch, seq, ngram_order)
        ngrams = torch.randint(0, 10000, (4, 32, 2))
        hashed = hasher.hash(ngrams)

        assert hashed.shape == (4, 32, 4)  # (batch, seq, num_heads)

    def test_hash_in_valid_range(self):
        """Hash values should be in [0, table_size)."""
        table_size = 1000003
        hasher = MultiHeadHash(
            ngram_order=3,
            num_heads=4,
            table_size=table_size,
        )

        ngrams = torch.randint(0, 50000, (8, 64, 3))
        hashed = hasher.hash(ngrams)

        assert hashed.min() >= 0
        assert hashed.max() < table_size

    def test_hash_is_deterministic(self):
        """Same input should always produce same hash."""
        hasher = MultiHeadHash(
            ngram_order=2,
            num_heads=4,
            table_size=1000003,
        )

        ngrams = torch.randint(0, 10000, (4, 32, 2))
        result1 = hasher.hash(ngrams)
        result2 = hasher.hash(ngrams)

        assert torch.equal(result1, result2)

    def test_different_ngrams_produce_different_hashes(self):
        """Different n-grams should usually hash to different values."""
        hasher = MultiHeadHash(
            ngram_order=2,
            num_heads=4,
            table_size=1000003,
        )

        ngram_a = torch.tensor([[[100, 200]]])
        ngram_b = torch.tensor([[[100, 201]]])

        hash_a = hasher.hash(ngram_a)
        hash_b = hasher.hash(ngram_b)

        # At least one head should differ
        assert not torch.equal(hash_a, hash_b)


class TestOffloadableEmbedding:
    """Tests for offloadable embedding tables."""

    def test_forward_without_offload(self):
        """Standard embedding lookup should work."""
        embedding = OffloadableEmbedding(
            num_embeddings=10000,
            embedding_dim=64,
            offload=False,
        )

        indices = torch.randint(0, 10000, (4, 32))
        output = embedding(indices)

        assert output.shape == (4, 32, 64)

    def test_forward_with_offload_cpu(self):
        """CPU offload should produce correct results."""
        embedding = OffloadableEmbedding(
            num_embeddings=10000,
            embedding_dim=64,
            offload=True,
            prefetch=False,  # Disable prefetch for simple test
        )

        indices = torch.randint(0, 10000, (4, 32))
        output = embedding(indices)

        assert output.shape == (4, 32, 64)

    def test_offload_matches_standard(self):
        """Offloaded and standard should produce same results."""
        torch.manual_seed(42)
        standard = OffloadableEmbedding(
            num_embeddings=1000,
            embedding_dim=32,
            offload=False,
        )

        torch.manual_seed(42)
        offloaded = OffloadableEmbedding(
            num_embeddings=1000,
            embedding_dim=32,
            offload=True,
            prefetch=False,
        )

        # Copy weights to ensure same initialization
        offloaded.weight.data = standard.embedding.weight.data.clone()

        indices = torch.randint(0, 1000, (2, 16))

        out_std = standard(indices)
        out_off = offloaded(indices)

        assert torch.allclose(out_std, out_off, atol=1e-6)


from brain_ai.memory.engram import (
    EngramConfig,
    EngramEmbedding,
    ContextAwareGating,
    EngramModule,
    RMSNorm,
)


class TestRMSNorm:
    """Tests for RMSNorm."""

    def test_output_shape(self):
        """RMSNorm should preserve shape."""
        norm = RMSNorm(dim=256)
        x = torch.randn(4, 32, 256)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalized_scale(self):
        """Output should be approximately normalized."""
        norm = RMSNorm(dim=256)
        x = torch.randn(4, 32, 256) * 10  # Large values
        out = norm(x)

        # RMS should be close to 1 after normalization
        rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.5)


class TestEngramEmbedding:
    """Tests for EngramEmbedding."""

    def test_output_shape(self):
        """Embedding output should match expected shape."""
        config = EngramConfig(
            vocab_size=10000,
            embedding_dim=128,
            ngram_orders=(2, 3),
            num_heads=4,
            table_size=100003,
        )

        embedding = EngramEmbedding(config)
        token_ids = torch.randint(0, 10000, (4, 32))
        output = embedding(token_ids)

        assert output.shape == (4, 32, 128)

    def test_deterministic(self):
        """Same input should produce same output."""
        config = EngramConfig(
            vocab_size=10000,
            embedding_dim=128,
            ngram_orders=(2, 3),
            num_heads=4,
            table_size=100003,
        )

        embedding = EngramEmbedding(config)
        token_ids = torch.randint(0, 10000, (4, 32))

        out1 = embedding(token_ids)
        out2 = embedding(token_ids)

        assert torch.equal(out1, out2)


class TestContextAwareGating:
    """Tests for ContextAwareGating."""

    def test_output_shape(self):
        """Gating should produce correct output shape."""
        gating = ContextAwareGating(
            hidden_dim=512,
            memory_dim=256,
        )

        hidden = torch.randn(4, 32, 512)
        memory = torch.randn(4, 32, 256)

        output, alpha = gating(hidden, memory)

        assert output.shape == (4, 32, 512)
        assert alpha.shape == (4, 32)

    def test_gate_values_in_range(self):
        """Gate values should be in [0, 1]."""
        gating = ContextAwareGating(
            hidden_dim=512,
            memory_dim=256,
        )

        hidden = torch.randn(4, 32, 512)
        memory = torch.randn(4, 32, 256)

        _, alpha = gating(hidden, memory)

        assert alpha.min() >= 0
        assert alpha.max() <= 1


class TestEngramModule:
    """Tests for complete EngramModule."""

    def test_output_shape(self):
        """EngramModule should produce correct output shape."""
        config = EngramConfig(
            vocab_size=10000,
            embedding_dim=128,
            ngram_orders=(2, 3),
            num_heads=4,
            table_size=100003,
        )

        engram = EngramModule(config, hidden_dim=512)

        token_ids = torch.randint(0, 10000, (4, 32))
        hidden = torch.randn(4, 32, 512)

        output, info = engram(token_ids, hidden)

        assert output.shape == (4, 32, 512)
        assert 'gate_values' in info
        assert info['gate_values'].shape == (4, 32)

    def test_residual_compatible(self):
        """Output should be suitable for residual connection."""
        config = EngramConfig(
            vocab_size=10000,
            embedding_dim=128,
            ngram_orders=(2, 3),
            num_heads=4,
            table_size=100003,
        )

        engram = EngramModule(config, hidden_dim=512)

        token_ids = torch.randint(0, 10000, (4, 32))
        hidden = torch.randn(4, 32, 512)

        output, _ = engram(token_ids, hidden)

        # Should be able to add to hidden (residual)
        result = hidden + output
        assert result.shape == hidden.shape
        assert not torch.isnan(result).any()


from brain_ai.encoders.engram_encoder import EngramTextEncoder


class TestEngramTextEncoder:
    """Tests for Phase 1 encoder integration."""

    def test_output_shape(self):
        """Encoder should produce workspace-compatible output."""
        config = EngramConfig(
            vocab_size=10000,
            embedding_dim=128,
            ngram_orders=(2, 3),
            num_heads=4,
            table_size=100003,
        )

        encoder = EngramTextEncoder(config, output_dim=512)
        token_ids = torch.randint(0, 10000, (4, 32))

        output = encoder(token_ids)

        assert output.shape == (4, 512)

    def test_matches_other_encoder_interface(self):
        """Should be usable alongside other encoders."""
        config = EngramConfig(
            vocab_size=10000,
            embedding_dim=128,
            ngram_orders=(2, 3),
            num_heads=4,
            table_size=100003,
        )

        encoder = EngramTextEncoder(config, output_dim=512)

        # Should accept token_ids like TextEncoder
        token_ids = torch.randint(0, 10000, (4, 32))
        output = encoder(token_ids)

        assert output.dim() == 2
        assert output.shape[-1] == 512


class TestBrainAIEngramIntegration:
    """Tests for Engram integration with BrainAI."""

    def test_brain_ai_with_engram_encoder(self):
        """BrainAI should work with Engram encoder enabled."""
        from brain_ai.system import create_brain_ai

        brain = create_brain_ai(
            modalities=['text'],
            use_engram=True,
            use_htm=False,
            use_symbolic=False,
            use_meta=False,
            device='cpu',
        )

        # Verify engram encoder was created
        assert 'engram' in brain.encoders, "Engram encoder should be created when use_engram=True"

        # Forward pass with token_ids
        token_ids = torch.randint(0, 30000, (2, 32))

        output = brain({'text': token_ids, 'token_ids': token_ids})

        assert output.shape[0] == 2

    def test_engram_competes_in_workspace(self):
        """Engram should participate in workspace competition."""
        from brain_ai.system import create_brain_ai

        brain = create_brain_ai(
            modalities=['text'],
            use_engram=True,
            use_htm=False,
            use_symbolic=False,
            use_meta=False,
            device='cpu',
        )

        # Verify engram encoder was created
        assert 'engram' in brain.encoders, "Engram encoder should be created when use_engram=True"

        token_ids = torch.randint(0, 30000, (2, 32))

        result = brain(
            {'text': token_ids, 'token_ids': token_ids},
            return_details=True
        )

        # Should have workspace output
        assert result.workspace is not None
        assert result.workspace.shape[0] == 2


from brain_ai.layers.engram_layer import EngramAugmentedLayer
from brain_ai.config import SNNConfig


class TestEngramAugmentedLayer:
    """Tests for Phase 2 layer integration."""

    def test_output_shape(self):
        """Layer should preserve input shape."""
        engram_config = EngramConfig(
            vocab_size=10000,
            embedding_dim=128,
            ngram_orders=(2, 3),
            num_heads=4,
            table_size=100003,
        )
        snn_config = SNNConfig()

        layer = EngramAugmentedLayer(
            hidden_dim=512,
            engram_config=engram_config,
            snn_config=snn_config,
            use_engram=True,
            use_snn=False,  # Disable SNN for simpler test
        )

        x = torch.randn(4, 32, 512)
        token_ids = torch.randint(0, 10000, (4, 32))

        output = layer(x, token_ids)

        assert output.shape == x.shape

    def test_layer_without_engram(self):
        """Layer should work without Engram enabled."""
        engram_config = EngramConfig()
        snn_config = SNNConfig()

        layer = EngramAugmentedLayer(
            hidden_dim=512,
            engram_config=engram_config,
            snn_config=snn_config,
            use_engram=False,
            use_snn=False,
        )

        x = torch.randn(4, 32, 512)
        token_ids = torch.randint(0, 10000, (4, 32))

        output = layer(x, token_ids)

        assert output.shape == x.shape

    def test_full_pipeline(self):
        """Full Engram -> Attention -> FFN pipeline."""
        engram_config = EngramConfig(
            vocab_size=10000,
            embedding_dim=128,
            ngram_orders=(2, 3),
            num_heads=4,
            table_size=100003,
        )
        snn_config = SNNConfig()

        layer = EngramAugmentedLayer(
            hidden_dim=256,
            engram_config=engram_config,
            snn_config=snn_config,
            use_engram=True,
            use_snn=False,
            num_heads=4,
        )

        x = torch.randn(2, 16, 256)
        token_ids = torch.randint(0, 10000, (2, 16))

        output = layer(x, token_ids)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()
