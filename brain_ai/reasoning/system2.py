"""
System 2 Reasoning Module

Implements dual-process theory of cognition:
- System 1: Fast, automatic, parallel processing
- System 2: Slow, deliberate, sequential reasoning

The module routes inputs between systems based on confidence:
- High confidence (>0.7): Use System 1 fast path
- Low confidence (<=0.7): Engage System 2 deliberation

This mirrors how humans effortlessly handle routine tasks but
slow down for novel or difficult problems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from .symbolic import SymbolicReasoner, SymbolicConfig


@dataclass
class System2Config:
    """Configuration for System 2 reasoning."""
    hidden_dim: int = 512
    confidence_threshold: float = 0.7
    max_iterations: int = 5
    min_iterations: int = 1

    # System 1 (fast)
    system1_hidden: int = 256
    system1_layers: int = 2

    # System 2 (deliberate)
    num_reasoning_steps: int = 3
    reasoning_hidden: int = 512

    # Metacognition
    use_metacognition: bool = True
    meta_hidden: int = 128


class System1Module(nn.Module):
    """
    System 1: Fast, automatic processing.

    Provides quick responses for familiar patterns.
    Low latency, parallel processing.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Confidence estimator
        self.confidence = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fast forward pass.

        Returns:
            output: Processed output
            confidence: Confidence in the output
        """
        output = self.network(x)
        confidence = self.confidence(x)
        return output, confidence


class System2Module(nn.Module):
    """
    System 2: Deliberate, sequential reasoning.

    Engages for novel or difficult problems.
    Multi-step iterative refinement with symbolic reasoning.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_reasoning_steps: int = 3,
        use_symbolic: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_reasoning_steps = num_reasoning_steps
        self.use_symbolic = use_symbolic

        # Working state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Reasoning GRU for iterative refinement
        self.reasoning_gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Symbolic reasoner
        if use_symbolic:
            self.symbolic = SymbolicReasoner(
                SymbolicConfig(hidden_dim=hidden_dim)
            )
        else:
            self.symbolic = None

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Confidence estimator (evolves during reasoning)
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Early stopping predictor
        self.stop_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def reason_step(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single deliberation step."""
        # GRU update
        new_state = self.reasoning_gru(context, state)

        # Symbolic reasoning if available
        if self.symbolic is not None:
            symbolic_out = self.symbolic(new_state, num_steps=1)
            new_state = new_state + 0.5 * symbolic_out['output']

        # Current confidence
        confidence = self.confidence_net(new_state)

        return new_state, confidence

    def forward(
        self,
        x: torch.Tensor,
        max_iterations: Optional[int] = None,
        return_trace: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Deliberate reasoning with iterative refinement.

        Args:
            x: Input (batch, input_dim)
            max_iterations: Override max reasoning steps
            return_trace: Whether to return reasoning trace

        Returns:
            Dict with output, confidence, num_iterations, and optionally trace
        """
        batch_size = x.shape[0]
        max_iter = max_iterations or self.num_reasoning_steps

        # Initialize state
        state = self.state_encoder(x)
        context = x

        trace = [state] if return_trace else None
        confidences = []

        # Iterative reasoning
        for i in range(max_iter):
            prev_state = state
            state, confidence = self.reason_step(state, context)
            confidences.append(confidence)

            if return_trace:
                trace.append(state)

            # Early stopping check
            combined = torch.cat([prev_state, state], dim=-1)
            should_stop = self.stop_predictor(combined)

            # Stop if converged (high confidence of stopping)
            if (should_stop > 0.8).all() and i >= 1:
                break

        # Project to output
        output = self.output_proj(state)
        final_confidence = confidences[-1]

        result = {
            'output': output,
            'confidence': final_confidence,
            'num_iterations': torch.tensor(i + 1),
            'state': state,
        }

        if return_trace:
            result['trace'] = torch.stack(trace, dim=1)
            result['confidence_trace'] = torch.stack(confidences, dim=1)

        return result


class MetacognitionModule(nn.Module):
    """
    Metacognition: Thinking about thinking.

    Monitors reasoning quality and decides when to
    switch between systems or request more processing.
    """

    def __init__(
        self,
        hidden_dim: int,
        meta_hidden: int = 128,
    ):
        super().__init__()

        # Uncertainty detector
        self.uncertainty = nn.Sequential(
            nn.Linear(hidden_dim, meta_hidden),
            nn.ReLU(),
            nn.Linear(meta_hidden, 1),
            nn.Sigmoid(),
        )

        # Novelty detector
        self.novelty = nn.Sequential(
            nn.Linear(hidden_dim, meta_hidden),
            nn.ReLU(),
            nn.Linear(meta_hidden, 1),
            nn.Sigmoid(),
        )

        # Effort predictor: how much System 2 is needed
        self.effort_predictor = nn.Sequential(
            nn.Linear(hidden_dim, meta_hidden),
            nn.ReLU(),
            nn.Linear(meta_hidden, 1),
            nn.Sigmoid(),
        )

        # System selector
        self.system_selector = nn.Sequential(
            nn.Linear(hidden_dim + 3, meta_hidden),
            nn.ReLU(),
            nn.Linear(meta_hidden, 2),  # System 1 vs System 2
        )

    def forward(
        self,
        x: torch.Tensor,
        system1_confidence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Metacognitive assessment.

        Returns:
            Dict with uncertainty, novelty, effort, system_choice
        """
        uncertainty = self.uncertainty(x)
        novelty = self.novelty(x)
        effort = self.effort_predictor(x)

        # Combine features for system selection
        if system1_confidence is not None:
            features = torch.cat([x, uncertainty, novelty, system1_confidence], dim=-1)
        else:
            features = torch.cat([x, uncertainty, novelty, effort], dim=-1)

        system_logits = self.system_selector(features)
        system_probs = F.softmax(system_logits, dim=-1)

        return {
            'uncertainty': uncertainty,
            'novelty': novelty,
            'effort': effort,
            'system_probs': system_probs,  # [P(System1), P(System2)]
        }


class DualProcessReasoner(nn.Module):
    """
    Dual-Process Reasoner combining System 1 and System 2.

    Routes inputs based on confidence and metacognitive assessment:
    - Simple/familiar -> System 1 (fast)
    - Complex/novel -> System 2 (deliberate)

    Args:
        config: System2Config with all parameters
    """

    def __init__(
        self,
        config: Optional[System2Config] = None,
        **kwargs,
    ):
        super().__init__()

        self.config = config or System2Config(**kwargs)

        # System 1: Fast path
        self.system1 = System1Module(
            input_dim=self.config.hidden_dim,
            output_dim=self.config.hidden_dim,
            hidden_dim=self.config.system1_hidden,
            num_layers=self.config.system1_layers,
        )

        # System 2: Slow path
        self.system2 = System2Module(
            input_dim=self.config.hidden_dim,
            output_dim=self.config.hidden_dim,
            hidden_dim=self.config.reasoning_hidden,
            num_reasoning_steps=self.config.num_reasoning_steps,
        )

        # Metacognition
        if self.config.use_metacognition:
            self.metacognition = MetacognitionModule(
                hidden_dim=self.config.hidden_dim,
                meta_hidden=self.config.meta_hidden,
            )
        else:
            self.metacognition = None

        # Output integration
        self.integration = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        force_system: Optional[int] = None,
        return_details: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Dual-process reasoning.

        Args:
            x: Input (batch, hidden_dim)
            force_system: Force use of system 1 or 2 (for debugging)
            return_details: Return detailed breakdown

        Returns:
            Dict with output, confidence, system_used, and optionally details
        """
        batch_size = x.shape[0]
        device = x.device

        # System 1: Always compute (fast)
        sys1_output, sys1_confidence = self.system1(x)

        # Metacognitive assessment
        if self.metacognition is not None:
            meta = self.metacognition(x, sys1_confidence)
            system_probs = meta['system_probs']
        else:
            # Default: use confidence threshold
            use_sys2 = (sys1_confidence < self.config.confidence_threshold).float()
            system_probs = torch.stack([1 - use_sys2, use_sys2], dim=-1).squeeze(-2)

        # Determine which system to use
        if force_system is not None:
            use_system2 = torch.full((batch_size,), force_system == 2, device=device)
        else:
            # Use System 2 if its probability is higher or confidence is low
            use_system2 = (
                (system_probs[:, 1] > system_probs[:, 0]) |
                (sys1_confidence.squeeze(-1) < self.config.confidence_threshold)
            )

        # System 2: Only compute if needed (selective)
        if use_system2.any():
            sys2_result = self.system2(x[use_system2])
            sys2_output = torch.zeros_like(sys1_output)
            sys2_confidence = torch.zeros_like(sys1_confidence)
            sys2_output[use_system2] = sys2_result['output']
            sys2_confidence[use_system2] = sys2_result['confidence']
        else:
            sys2_output = torch.zeros_like(sys1_output)
            sys2_confidence = torch.zeros_like(sys1_confidence)

        # Soft blending based on system probabilities
        sys1_weight = system_probs[:, 0:1]
        sys2_weight = system_probs[:, 1:2]

        # Weighted combination
        blended = sys1_weight * sys1_output + sys2_weight * sys2_output

        # Alternative: Hard selection
        hard_output = torch.where(
            use_system2.unsqueeze(-1),
            sys2_output,
            sys1_output
        )

        # Use blended for training, hard for inference
        output = blended if self.training else hard_output

        # Combined confidence
        confidence = sys1_weight * sys1_confidence + sys2_weight * sys2_confidence

        result = {
            'output': output,
            'confidence': confidence,
            'system_used': use_system2.float(),
            'sys1_output': sys1_output,
            'sys2_output': sys2_output,
        }

        if return_details:
            result['sys1_confidence'] = sys1_confidence
            result['sys2_confidence'] = sys2_confidence
            result['system_probs'] = system_probs
            if self.metacognition is not None:
                result['metacognition'] = meta

        return result


class AdaptiveReasoner(DualProcessReasoner):
    """
    Adaptive Reasoner that learns when to engage each system.

    Extends DualProcessReasoner with:
    1. Learning from feedback about system choice quality
    2. Adapting confidence thresholds based on performance
    3. Tracking reasoning statistics
    """

    def __init__(
        self,
        config: Optional[System2Config] = None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)

        # Adaptive threshold
        self.register_buffer(
            'confidence_threshold',
            torch.tensor(self.config.confidence_threshold)
        )

        # Statistics tracking
        self.register_buffer('sys1_correct', torch.tensor(0.0))
        self.register_buffer('sys1_total', torch.tensor(0.0))
        self.register_buffer('sys2_correct', torch.tensor(0.0))
        self.register_buffer('sys2_total', torch.tensor(0.0))

    def update_stats(
        self,
        system_used: torch.Tensor,
        was_correct: torch.Tensor,
    ):
        """Update statistics based on feedback."""
        sys1_mask = ~system_used.bool()
        sys2_mask = system_used.bool()

        self.sys1_total += sys1_mask.sum()
        self.sys1_correct += (was_correct & sys1_mask).sum()

        self.sys2_total += sys2_mask.sum()
        self.sys2_correct += (was_correct & sys2_mask).sum()

    def adapt_threshold(self, target_accuracy: float = 0.9):
        """Adapt confidence threshold based on System 1 accuracy."""
        if self.sys1_total > 100:  # Need enough samples
            sys1_accuracy = self.sys1_correct / (self.sys1_total + 1e-8)

            if sys1_accuracy < target_accuracy:
                # System 1 too inaccurate, raise threshold
                self.confidence_threshold = torch.clamp(
                    self.confidence_threshold + 0.01,
                    max=0.95
                )
            elif sys1_accuracy > target_accuracy + 0.05:
                # System 1 very accurate, lower threshold
                self.confidence_threshold = torch.clamp(
                    self.confidence_threshold - 0.01,
                    min=0.5
                )


# Factory function
def create_dual_process_reasoner(
    hidden_dim: int = 512,
    confidence_threshold: float = 0.7,
    max_iterations: int = 5,
    use_metacognition: bool = True,
    adaptive: bool = False,
    **kwargs,
) -> DualProcessReasoner:
    """Create dual-process reasoner with specified configuration."""
    config = System2Config(
        hidden_dim=hidden_dim,
        confidence_threshold=confidence_threshold,
        max_iterations=max_iterations,
        use_metacognition=use_metacognition,
        **kwargs,
    )

    if adaptive:
        return AdaptiveReasoner(config)
    else:
        return DualProcessReasoner(config)
