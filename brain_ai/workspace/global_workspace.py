"""
Global Workspace Module

Implements Global Workspace Theory (GWT) for multi-modal integration.

The Global Workspace serves as an "information broadcast" system where:
1. Specialist modules (encoders) compete for workspace access
2. Winners broadcast their information to all other modules
3. Attention mechanisms control what enters the workspace
4. Working memory maintains temporal context

This mirrors prefrontal cortex function and consciousness theories.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

from .working_memory import WorkingMemory, create_working_memory


@dataclass
class GlobalWorkspaceConfig:
    """Configuration for Global Workspace."""
    workspace_dim: int = 512
    num_heads: int = 8
    capacity_limit: int = 7  # Miller's Law: 7 +/- 2
    dropout: float = 0.1

    # Working memory
    memory_hidden_dim: int = 512
    memory_mode: str = "cfc"

    # Competition
    competition_temperature: float = 1.0
    min_attention: float = 0.01  # Minimum attention to prevent complete suppression


class ModalityProjection(nn.Module):
    """
    Projection layer for a single modality into workspace.

    Projects modality-specific features into common workspace dimension.
    """

    def __init__(
        self,
        input_dim: int,
        workspace_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, workspace_dim),
            nn.LayerNorm(workspace_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(workspace_dim, workspace_dim),
        )

        # Salience predictor: how important is this input?
        self.salience = nn.Sequential(
            nn.Linear(input_dim, workspace_dim // 2),
            nn.ReLU(),
            nn.Linear(workspace_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project modality input to workspace.

        Returns:
            projected: Features in workspace dimension
            salience: Importance score for competition
        """
        projected = self.projection(x)
        salience = self.salience(x)
        return projected, salience


class AttentionCompetition(nn.Module):
    """
    Attention-based competition for workspace access.

    Multiple modalities compete for limited workspace capacity.
    Uses multi-head attention with top-K gating.
    """

    def __init__(
        self,
        workspace_dim: int = 512,
        num_heads: int = 8,
        capacity_limit: int = 7,
        temperature: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.workspace_dim = workspace_dim
        self.num_heads = num_heads
        self.capacity_limit = capacity_limit
        self.temperature = temperature

        # Multi-head self-attention for competition
        self.attention = nn.MultiheadAttention(
            embed_dim=workspace_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Gating network for capacity control
        self.gate = nn.Sequential(
            nn.Linear(workspace_dim, workspace_dim // 2),
            nn.ReLU(),
            nn.Linear(workspace_dim // 2, 1),
        )

        # Output normalization
        self.norm = nn.LayerNorm(workspace_dim)

    def forward(
        self,
        features: torch.Tensor,
        saliences: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compete for workspace access.

        Args:
            features: Modality features (batch, num_modalities, workspace_dim)
            saliences: Importance scores (batch, num_modalities, 1)
            mask: Optional attention mask

        Returns:
            winners: Features that won access (batch, k, workspace_dim)
            attention_weights: Competition results (batch, num_modalities)
        """
        batch_size, num_items, _ = features.shape

        # Self-attention among modalities
        # Each modality attends to all others
        attended, raw_weights = self.attention(
            features, features, features,
            attn_mask=mask,
            need_weights=True,
        )

        # Combine attention weights with salience
        gate_scores = self.gate(attended).squeeze(-1)  # (batch, num_items)
        combined_scores = gate_scores + saliences.squeeze(-1)

        # Apply temperature
        combined_scores = combined_scores / self.temperature

        # Soft top-K gating
        # Sort by score and create soft mask
        attention_weights = F.softmax(combined_scores, dim=-1)

        # Hard top-K selection for capacity limit
        if num_items > self.capacity_limit:
            _, top_indices = torch.topk(
                attention_weights,
                self.capacity_limit,
                dim=-1
            )

            # Create selection mask
            selection_mask = torch.zeros_like(attention_weights)
            selection_mask.scatter_(1, top_indices, 1.0)

            # Apply mask while maintaining gradients
            attention_weights = attention_weights * selection_mask
            attention_weights = attention_weights / (
                attention_weights.sum(dim=-1, keepdim=True) + 1e-8
            )

        # Weight features by attention
        winners = attended * attention_weights.unsqueeze(-1)
        winners = self.norm(winners)

        return winners, attention_weights


class InformationBroadcast(nn.Module):
    """
    Broadcast workspace content back to specialists.

    After competition, winning information is broadcast to all
    modules, enabling global information sharing.
    """

    def __init__(
        self,
        workspace_dim: int,
        modality_dims: Dict[str, int],
        dropout: float = 0.1,
    ):
        super().__init__()

        self.workspace_dim = workspace_dim
        self.modality_dims = modality_dims

        # Broadcast projections for each modality
        self.broadcast_projections = nn.ModuleDict()
        for name, dim in modality_dims.items():
            self.broadcast_projections[name] = nn.Sequential(
                nn.Linear(workspace_dim, workspace_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(workspace_dim, dim),
            )

    def forward(
        self,
        workspace_content: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Broadcast workspace to all modalities.

        Args:
            workspace_content: Current workspace content (batch, workspace_dim)

        Returns:
            Dict of broadcast signals for each modality
        """
        broadcasts = {}
        for name, projection in self.broadcast_projections.items():
            broadcasts[name] = projection(workspace_content)
        return broadcasts


class GlobalWorkspace(nn.Module):
    """
    Complete Global Workspace for multi-modal integration.

    Combines:
    1. Modality projections into common space
    2. Attention-based competition for workspace access
    3. Working memory for temporal context
    4. Information broadcast back to specialists

    Args:
        config: GlobalWorkspaceConfig with all parameters
        modality_dims: Dict mapping modality names to their dimensions
    """

    def __init__(
        self,
        config: Optional[GlobalWorkspaceConfig] = None,
        modality_dims: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        super().__init__()

        self.config = config or GlobalWorkspaceConfig(**kwargs)

        # Default modality dimensions
        self.modality_dims = modality_dims or {
            'vision': 512,
            'text': 512,
            'audio': 512,
            'sensors': 512,
        }

        # Create modality projections
        self.projections = nn.ModuleDict()
        for name, dim in self.modality_dims.items():
            self.projections[name] = ModalityProjection(
                input_dim=dim,
                workspace_dim=self.config.workspace_dim,
                dropout=self.config.dropout,
            )

        # Competition mechanism
        self.competition = AttentionCompetition(
            workspace_dim=self.config.workspace_dim,
            num_heads=self.config.num_heads,
            capacity_limit=self.config.capacity_limit,
            temperature=self.config.competition_temperature,
            dropout=self.config.dropout,
        )

        # Working memory
        self.working_memory = create_working_memory(
            input_dim=self.config.workspace_dim,
            hidden_dim=self.config.memory_hidden_dim,
            output_dim=self.config.workspace_dim,
            mode=self.config.memory_mode,
        )

        # Information broadcast
        self.broadcast = InformationBroadcast(
            workspace_dim=self.config.workspace_dim,
            modality_dims=self.modality_dims,
            dropout=self.config.dropout,
        )

        # Integration layer
        self.integration = nn.Sequential(
            nn.Linear(self.config.workspace_dim * 2, self.config.workspace_dim),
            nn.LayerNorm(self.config.workspace_dim),
            nn.ReLU(),
            nn.Linear(self.config.workspace_dim, self.config.workspace_dim),
        )

        # Context from previous timestep
        self.register_buffer('prev_context', None)

    def reset_state(self):
        """Reset workspace state."""
        self.working_memory.reset_state()
        self.prev_context = None

    def forward(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Process multi-modal inputs through global workspace.

        Args:
            modality_inputs: Dict mapping modality names to features
                Each tensor has shape (batch, modality_dim)
            return_attention: Whether to return attention weights

        Returns:
            Dict with:
                - workspace: Integrated workspace representation (batch, workspace_dim)
                - broadcasts: Dict of broadcast signals per modality
                - attention: Competition attention weights (if return_attention)
                - memory_output: Working memory output
        """
        batch_size = next(iter(modality_inputs.values())).shape[0]
        device = next(iter(modality_inputs.values())).device

        # 1. Project all modalities to workspace dimension
        projected = []
        saliences = []
        modality_names = []

        for name, features in modality_inputs.items():
            if name in self.projections:
                proj, sal = self.projections[name](features)
                projected.append(proj)
                saliences.append(sal)
                modality_names.append(name)

        if len(projected) == 0:
            raise ValueError("No valid modality inputs provided")

        # Stack into tensors
        # (batch, num_modalities, workspace_dim)
        projected = torch.stack(projected, dim=1)
        saliences = torch.stack(saliences, dim=1)

        # 2. Competition for workspace access
        winners, attention_weights = self.competition(
            projected, saliences
        )

        # 3. Aggregate winning content
        # Weighted sum of winners
        workspace_content = winners.sum(dim=1)  # (batch, workspace_dim)

        # 4. Integrate with previous context via working memory
        if self.prev_context is not None:
            # Combine current with previous
            combined = torch.cat([workspace_content, self.prev_context], dim=-1)
            integrated = self.integration(combined)
        else:
            integrated = workspace_content

        # 5. Update working memory
        memory_result = self.working_memory(integrated)
        memory_output = memory_result['output']

        # 6. Update context for next step
        self.prev_context = memory_output.detach()

        # 7. Broadcast to all modalities
        broadcasts = self.broadcast(memory_output)

        # Build output
        output = {
            'workspace': memory_output,
            'broadcasts': broadcasts,
            'memory_output': memory_result,
            'modality_names': modality_names,
        }

        if return_attention:
            # Map attention weights to modality names
            attention_dict = {
                name: attention_weights[:, i]
                for i, name in enumerate(modality_names)
            }
            output['attention'] = attention_dict

        return output

    def get_workspace_state(self) -> Dict[str, torch.Tensor]:
        """Get current workspace state for inspection."""
        return {
            'prev_context': self.prev_context,
            'memory_buffer': self.working_memory.memory_buffer,
        }


class GlobalWorkspaceWithHTM(GlobalWorkspace):
    """
    Global Workspace with HTM integration.

    Extends the base workspace to incorporate HTM anomaly detection
    and temporal predictions into the workspace dynamics.
    """

    def __init__(
        self,
        config: Optional[GlobalWorkspaceConfig] = None,
        modality_dims: Optional[Dict[str, int]] = None,
        htm_layer=None,
        **kwargs,
    ):
        super().__init__(config, modality_dims, **kwargs)

        self.htm_layer = htm_layer

        # Anomaly-based attention modulation
        self.anomaly_gate = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Process with HTM integration."""

        # Get base workspace output
        output = super().forward(modality_inputs, return_attention)

        # Process through HTM if available
        if self.htm_layer is not None:
            workspace = output['workspace']
            htm_result = self.htm_layer(workspace)

            # Modulate based on anomaly
            anomaly = htm_result['anomaly']
            if anomaly.dim() == 0:
                anomaly = anomaly.unsqueeze(0)
            if anomaly.dim() == 1:
                anomaly = anomaly.unsqueeze(-1)

            anomaly_gate = self.anomaly_gate(anomaly)

            # High anomaly = increase attention/arousal
            # This makes the system more receptive to novel inputs
            output['workspace'] = output['workspace'] * (1 + anomaly_gate)
            output['htm'] = htm_result

        return output


# Factory function
def create_global_workspace(
    workspace_dim: int = 512,
    modality_dims: Optional[Dict[str, int]] = None,
    num_heads: int = 8,
    capacity_limit: int = 7,
    memory_mode: str = "auto",
    use_htm: bool = False,
    htm_layer=None,
    **kwargs,
) -> GlobalWorkspace:
    """
    Create Global Workspace with specified configuration.

    Args:
        workspace_dim: Dimension of workspace representation
        modality_dims: Dict of modality dimensions
        num_heads: Number of attention heads
        capacity_limit: Maximum items in working memory
        memory_mode: Working memory backend ('cfc', 'ltc', 'gru', 'auto')
        use_htm: Whether to integrate HTM
        htm_layer: Pre-configured HTM layer
    """
    config = GlobalWorkspaceConfig(
        workspace_dim=workspace_dim,
        num_heads=num_heads,
        capacity_limit=capacity_limit,
        memory_mode=memory_mode,
        **kwargs,
    )

    if use_htm:
        return GlobalWorkspaceWithHTM(
            config=config,
            modality_dims=modality_dims,
            htm_layer=htm_layer,
        )
    else:
        return GlobalWorkspace(
            config=config,
            modality_dims=modality_dims,
        )
