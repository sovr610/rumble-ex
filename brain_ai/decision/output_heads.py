"""
Output Heads Module

Provides different output capabilities for the brain-inspired AI system:
- Classification: Discrete class predictions
- Text Generation: Autoregressive text decoding
- Continuous Control: Gaussian policy for motor control

Each head takes workspace representation as input and produces
task-specific outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass
import math


@dataclass
class OutputHeadsConfig:
    """Configuration for output heads."""
    input_dim: int = 512  # From workspace

    # Classification
    num_classes: int = 10
    classifier_hidden: int = 256

    # Text generation
    vocab_size: int = 30000
    text_hidden_dim: int = 512
    text_num_layers: int = 2
    text_num_heads: int = 8
    max_seq_length: int = 512

    # Continuous control
    control_dim: int = 6  # e.g., 6-DOF robot arm
    control_hidden: int = 256

    # General
    dropout: float = 0.1


class ClassificationHead(nn.Module):
    """
    Classification output head.

    Produces discrete class predictions from workspace representation.
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_classes: int = 10,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Confidence estimation
        self.confidence = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_confidence: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Classify input.

        Args:
            x: Workspace representation (batch, input_dim)
            return_confidence: Whether to return confidence estimate

        Returns:
            Dict with logits, probs, predictions, and optionally confidence
        """
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=-1)
        predictions = probs.argmax(dim=-1)

        output = {
            'logits': logits,
            'probs': probs,
            'predictions': predictions,
        }

        if return_confidence:
            confidence = self.confidence(x)
            output['confidence'] = confidence

        return output


class TextDecoderHead(nn.Module):
    """
    Autoregressive text generation head.

    Uses Transformer decoder to generate text from workspace representation.
    """

    def __init__(
        self,
        input_dim: int = 512,
        vocab_size: int = 30000,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        max_seq_length: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_dim)

        # Project workspace to decoder dimension
        self.workspace_proj = nn.Linear(input_dim, hidden_dim)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        # Causal mask cache
        self.register_buffer(
            'causal_mask',
            self._generate_causal_mask(max_seq_length)
        )

    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(
        self,
        workspace: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            workspace: Workspace representation (batch, input_dim)
            target_tokens: Target token ids (batch, seq_len) for teacher forcing

        Returns:
            Dict with logits and loss (if targets provided)
        """
        batch_size = workspace.shape[0]
        device = workspace.device

        # Project workspace to memory for cross-attention
        memory = self.workspace_proj(workspace).unsqueeze(1)  # (batch, 1, hidden)

        if target_tokens is None:
            # Just return memory encoding for generation
            return {'memory': memory}

        seq_len = target_tokens.shape[1]

        # Embed tokens
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        token_emb = self.token_embedding(target_tokens)
        pos_emb = self.position_embedding(positions)
        tgt = token_emb + pos_emb

        # Get causal mask
        causal_mask = self.causal_mask[:seq_len, :seq_len]

        # Decode
        decoded = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
        )

        # Project to vocabulary
        logits = self.output_proj(decoded)

        output = {'logits': logits}

        return output

    @torch.no_grad()
    def generate(
        self,
        workspace: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        start_token_id: int = 1,  # BOS token
        end_token_id: int = 2,  # EOS token
    ) -> torch.Tensor:
        """
        Autoregressively generate text.

        Args:
            workspace: Workspace representation (batch, input_dim)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus (top-p) filtering
            start_token_id: Start token id
            end_token_id: End token id

        Returns:
            generated_ids: Generated token ids (batch, seq_len)
        """
        batch_size = workspace.shape[0]
        device = workspace.device

        # Project workspace to memory
        memory = self.workspace_proj(workspace).unsqueeze(1)

        # Start with start token
        generated = torch.full(
            (batch_size, 1),
            start_token_id,
            dtype=torch.long,
            device=device
        )

        for _ in range(max_length - 1):
            seq_len = generated.shape[1]

            # Embed current sequence
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            token_emb = self.token_embedding(generated)
            pos_emb = self.position_embedding(positions)
            tgt = token_emb + pos_emb

            # Get causal mask
            causal_mask = self.causal_mask[:seq_len, :seq_len]

            # Decode
            decoded = self.decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=causal_mask,
            )

            # Get next token logits
            next_logits = self.output_proj(decoded[:, -1, :])

            # Apply temperature
            next_logits = next_logits / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][:, -1:]
                next_logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

            # Check for end token
            if (next_token == end_token_id).all():
                break

        return generated


class ContinuousControlHead(nn.Module):
    """
    Continuous control output head.

    Outputs Gaussian distribution over continuous actions
    for motor control tasks.
    """

    def __init__(
        self,
        input_dim: int = 512,
        control_dim: int = 6,
        hidden_dim: int = 256,
        min_std: float = 0.01,
        max_std: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.control_dim = control_dim
        self.min_std = min_std
        self.max_std = max_std

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Mean and log std outputs
        self.mu_head = nn.Linear(hidden_dim, control_dim)
        self.log_std_head = nn.Linear(hidden_dim, control_dim)

        # Value estimate (for actor-critic)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute control action.

        Args:
            x: Workspace representation (batch, input_dim)
            deterministic: If True, return mean action

        Returns:
            Dict with action, mu, std, log_prob, value
        """
        h = self.encoder(x)

        # Get distribution parameters
        mu = self.mu_head(h)
        log_std = self.log_std_head(h)

        # Bound standard deviation
        std = torch.clamp(
            torch.exp(log_std),
            self.min_std,
            self.max_std
        )

        # Sample action
        if deterministic:
            action = mu
        else:
            eps = torch.randn_like(mu)
            action = mu + eps * std

        # Compute log probability
        log_prob = -0.5 * (
            ((action - mu) / std) ** 2 +
            2 * torch.log(std) +
            math.log(2 * math.pi)
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Value estimate
        value = self.value_head(h)

        return {
            'action': action,
            'mu': mu,
            'std': std,
            'log_prob': log_prob,
            'value': value,
        }

    def evaluate_actions(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate given actions (for policy gradient).

        Args:
            x: Workspace representation (batch, input_dim)
            actions: Actions to evaluate (batch, control_dim)

        Returns:
            Dict with log_prob, entropy, value
        """
        h = self.encoder(x)

        mu = self.mu_head(h)
        log_std = self.log_std_head(h)
        std = torch.clamp(torch.exp(log_std), self.min_std, self.max_std)

        # Log probability of given actions
        log_prob = -0.5 * (
            ((actions - mu) / std) ** 2 +
            2 * torch.log(std) +
            math.log(2 * math.pi)
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Entropy
        entropy = 0.5 * (1 + math.log(2 * math.pi) + 2 * log_std).sum(dim=-1)

        # Value
        value = self.value_head(h)

        return {
            'log_prob': log_prob,
            'entropy': entropy,
            'value': value,
        }


class DecisionHeads(nn.Module):
    """
    Combined output heads for multi-task outputs.

    Provides unified interface to multiple output types.
    """

    def __init__(
        self,
        config: Optional[OutputHeadsConfig] = None,
        **kwargs,
    ):
        super().__init__()

        self.config = config or OutputHeadsConfig(**kwargs)

        # Classification head
        self.classifier = ClassificationHead(
            input_dim=self.config.input_dim,
            num_classes=self.config.num_classes,
            hidden_dim=self.config.classifier_hidden,
            dropout=self.config.dropout,
        )

        # Text generation head
        self.text_decoder = TextDecoderHead(
            input_dim=self.config.input_dim,
            vocab_size=self.config.vocab_size,
            hidden_dim=self.config.text_hidden_dim,
            num_layers=self.config.text_num_layers,
            num_heads=self.config.text_num_heads,
            max_seq_length=self.config.max_seq_length,
            dropout=self.config.dropout,
        )

        # Continuous control head
        self.control = ContinuousControlHead(
            input_dim=self.config.input_dim,
            control_dim=self.config.control_dim,
            hidden_dim=self.config.control_hidden,
            dropout=self.config.dropout,
        )

    def classify(
        self,
        workspace: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Produce classification output."""
        return self.classifier(workspace, **kwargs)

    def generate_text(
        self,
        workspace: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
        max_length: int = 100,
        **kwargs,
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Generate text or compute training loss."""
        if target_tokens is not None:
            return self.text_decoder(workspace, target_tokens)
        else:
            return self.text_decoder.generate(workspace, max_length, **kwargs)

    def control_action(
        self,
        workspace: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Produce continuous control action."""
        return self.control(workspace, **kwargs)

    def forward(
        self,
        workspace: torch.Tensor,
        task: str = "classify",
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with task selection.

        Args:
            workspace: Workspace representation
            task: 'classify', 'generate', or 'control'
            **kwargs: Task-specific arguments
        """
        if task == "classify":
            return self.classify(workspace, **kwargs)
        elif task == "generate":
            return self.generate_text(workspace, **kwargs)
        elif task == "control":
            return self.control_action(workspace, **kwargs)
        else:
            raise ValueError(f"Unknown task: {task}")


# Factory function
def create_decision_heads(
    input_dim: int = 512,
    num_classes: int = 10,
    vocab_size: int = 30000,
    control_dim: int = 6,
    **kwargs,
) -> DecisionHeads:
    """Create decision heads with specified configuration."""
    config = OutputHeadsConfig(
        input_dim=input_dim,
        num_classes=num_classes,
        vocab_size=vocab_size,
        control_dim=control_dim,
        **kwargs,
    )
    return DecisionHeads(config)
