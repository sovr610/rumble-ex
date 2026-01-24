"""
Working Memory Module

Implements working memory using Liquid Neural Networks (Neural Circuit Policies).
Uses ncps library for CfC (Closed-form Continuous-time) and LTC (Liquid Time-Constant) networks.

Working memory maintains temporal state and integrates information over time,
similar to prefrontal cortex function.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

# Try to import ncps
try:
    from ncps.torch import CfC, LTC
    from ncps.wirings import AutoNCP, FullyConnected
    NCPS_AVAILABLE = True
except ImportError:
    NCPS_AVAILABLE = False


@dataclass
class WorkingMemoryConfig:
    """Configuration for working memory."""
    input_dim: int = 512
    hidden_dim: int = 512
    output_dim: int = 512
    mode: str = "cfc"  # cfc, ltc, or gru (fallback)
    num_units: int = 64  # For NCP wiring
    sparsity: float = 0.5  # Connection sparsity


class GRUWorkingMemory(nn.Module):
    """
    GRU-based working memory fallback.

    Used when ncps is not available. Provides similar
    temporal integration capabilities with standard GRU.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.hidden_state = None

    def reset_state(self):
        """Reset hidden state."""
        self.hidden_state = None

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input through working memory.

        Args:
            x: Input tensor (batch, features) or (batch, seq, features)
            state: Optional hidden state override

        Returns:
            output: Processed output (batch, output_dim)
            state: Updated hidden state
        """
        if state is not None:
            self.hidden_state = state

        # Handle input dimensions
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        # Forward through GRU
        output, self.hidden_state = self.gru(x, self.hidden_state)

        # Take last timestep and project
        output = self.output_proj(output[:, -1, :])

        return output, self.hidden_state


class LiquidWorkingMemory(nn.Module):
    """
    Liquid Neural Network working memory using ncps.

    Implements continuous-time dynamics that naturally handle
    irregular time intervals and temporal dependencies.

    CfC (Closed-form Continuous-time) provides:
    - Efficient closed-form solution
    - Bounded gradients
    - Natural handling of variable time steps

    LTC (Liquid Time-Constant) provides:
    - Biologically-inspired dynamics
    - Adaptive time constants
    - Better for very long sequences
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 512,
        mode: str = "cfc",
        num_units: int = 64,
        sparsity: float = 0.5,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.mode = mode

        if not NCPS_AVAILABLE:
            raise ImportError(
                "ncps package required for LiquidWorkingMemory. "
                "Install with: pip install ncps"
            )

        # Create wiring
        # AutoNCP requires output_size < num_units - 2
        # We use a smaller internal output size and project to hidden_dim
        ncp_output_size = min(hidden_dim, num_units - 4)  # Leave margin for NCP constraint
        if ncp_output_size < 1:
            ncp_output_size = num_units // 2  # Fallback to half units
        
        # Ensure num_units is large enough
        effective_units = max(num_units, ncp_output_size + 4)
        
        wiring = AutoNCP(
            units=effective_units,
            output_size=ncp_output_size,
            sparsity_level=sparsity,
        )
        
        self.ncp_output_size = ncp_output_size

        # Create liquid network
        if mode == "cfc":
            self.liquid = CfC(
                input_size=input_dim,
                units=wiring,
                return_sequences=False,
                batch_first=True,
            )
        elif mode == "ltc":
            self.liquid = LTC(
                input_size=input_dim,
                units=wiring,
                return_sequences=False,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'cfc' or 'ltc'")

        # Projection from NCP output to hidden_dim, then to output_dim
        self.hidden_proj = nn.Linear(ncp_output_size, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.hidden_state = None

    def reset_state(self):
        """Reset hidden state."""
        self.hidden_state = None

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        timespans: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input through liquid working memory.

        Args:
            x: Input tensor (batch, features) or (batch, seq, features)
            state: Optional hidden state override
            timespans: Optional time intervals for continuous-time mode

        Returns:
            output: Processed output (batch, output_dim)
            state: Updated hidden state
        """
        if state is not None:
            self.hidden_state = state

        # Handle input dimensions
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        # Forward through liquid network
        if self.mode == "cfc":
            output, self.hidden_state = self.liquid(
                x,
                hx=self.hidden_state,
                timespans=timespans,
            )
        else:
            output, self.hidden_state = self.liquid(
                x,
                hx=self.hidden_state,
            )

        # Project from NCP output to hidden_dim, then to output_dim
        output = self.hidden_proj(output)
        output = self.output_proj(output)

        return output, self.hidden_state


class WorkingMemory(nn.Module):
    """
    Working Memory module with automatic backend selection.

    Uses Liquid Neural Networks (CfC/LTC) if ncps is available,
    otherwise falls back to GRU-based implementation.

    Working memory serves to:
    1. Maintain task-relevant information over time
    2. Integrate multi-modal inputs
    3. Support working memory capacity limits (~7 items)
    4. Enable temporal reasoning
    """

    def __init__(
        self,
        config: Optional[WorkingMemoryConfig] = None,
        **kwargs,
    ):
        super().__init__()

        self.config = config or WorkingMemoryConfig(**kwargs)

        # Select backend
        if NCPS_AVAILABLE and self.config.mode in ("cfc", "ltc"):
            self.backend = LiquidWorkingMemory(
                input_dim=self.config.input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=self.config.output_dim,
                mode=self.config.mode,
                num_units=self.config.num_units,
                sparsity=self.config.sparsity,
            )
            self.backend_type = self.config.mode
        else:
            self.backend = GRUWorkingMemory(
                input_dim=self.config.input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=self.config.output_dim,
            )
            self.backend_type = "gru"

        # Memory buffer for capacity-limited storage
        # Stores up to K items (Miller's 7 +/- 2)
        self.register_buffer('memory_buffer', None)
        self.capacity = 7

    def reset_state(self):
        """Reset working memory state."""
        self.backend.reset_state()
        self.memory_buffer = None

    def update_buffer(self, item: torch.Tensor) -> torch.Tensor:
        """
        Add item to memory buffer with capacity limit.

        Implements a FIFO queue with limited capacity.
        """
        if self.memory_buffer is None:
            self.memory_buffer = item.unsqueeze(1)
        else:
            self.memory_buffer = torch.cat(
                [self.memory_buffer, item.unsqueeze(1)],
                dim=1
            )
            # Enforce capacity limit
            if self.memory_buffer.shape[1] > self.capacity:
                self.memory_buffer = self.memory_buffer[:, -self.capacity:]

        return self.memory_buffer

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        timespans: Optional[torch.Tensor] = None,
        update_buffer: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Process input through working memory.

        Args:
            x: Input tensor (batch, features)
            state: Optional hidden state override
            timespans: Time intervals (for CfC mode)
            update_buffer: Whether to update memory buffer

        Returns:
            Dict with:
                - output: Processed output
                - state: Hidden state
                - buffer: Current memory buffer contents
        """
        # Process through backend
        if self.backend_type == "cfc" and timespans is not None:
            output, new_state = self.backend(x, state, timespans)
        else:
            output, new_state = self.backend(x, state)

        # Update memory buffer
        if update_buffer:
            buffer = self.update_buffer(output)
        else:
            buffer = self.memory_buffer

        return {
            'output': output,
            'state': new_state,
            'buffer': buffer,
        }

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """
        Retrieve from memory buffer using attention.

        Args:
            query: Query vector (batch, features)

        Returns:
            Retrieved content via attention-weighted sum
        """
        if self.memory_buffer is None or self.memory_buffer.shape[1] == 0:
            return query  # Return query if buffer empty

        # Compute attention scores
        # query: (batch, features), buffer: (batch, items, features)
        scores = torch.bmm(
            self.memory_buffer,
            query.unsqueeze(-1)
        ).squeeze(-1)  # (batch, items)

        attention = torch.softmax(scores, dim=-1)

        # Weighted retrieval
        retrieved = torch.bmm(
            attention.unsqueeze(1),
            self.memory_buffer
        ).squeeze(1)  # (batch, features)

        return retrieved


# Factory function
def create_working_memory(
    input_dim: int = 512,
    hidden_dim: int = 512,
    output_dim: int = 512,
    mode: str = "auto",
    **kwargs,
) -> WorkingMemory:
    """
    Create working memory with automatic backend selection.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        mode: 'cfc', 'ltc', 'gru', or 'auto'
        **kwargs: Additional config parameters
    """
    if mode == "auto":
        mode = "cfc" if NCPS_AVAILABLE else "gru"

    config = WorkingMemoryConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        mode=mode,
        **kwargs,
    )

    return WorkingMemory(config)
