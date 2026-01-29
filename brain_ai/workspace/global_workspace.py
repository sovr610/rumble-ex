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
            # Check batch size compatibility - reset if mismatch
            if self.prev_context.shape[0] != workspace_content.shape[0]:
                self.prev_context = None
        
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


# =============================================================================
# IMPROVED GLOBAL WORKSPACE: Selection-Broadcast with Ignition (2025)
# =============================================================================

@dataclass
class SelectionBroadcastConfig:
    """Configuration for Selection-Broadcast workspace."""
    workspace_dim: int = 1024  # Increased default
    num_heads: int = 16
    capacity_limit: int = 7
    dropout: float = 0.1
    
    # Selection parameters
    ignition_threshold: float = 0.3  # Threshold for global ignition
    selection_rounds: int = 3  # Iterative selection rounds
    
    # Broadcast parameters
    broadcast_iterations: int = 2  # Broadcast refinement iterations
    broadcast_decay: float = 0.9  # Temporal decay of broadcast
    
    # Working memory
    memory_hidden_dim: int = 1024
    memory_mode: str = "cfc"
    
    # Competition dynamics
    competition_temperature: float = 0.5  # Lower = sharper competition
    min_attention: float = 0.01
    
    # Meta-cognition
    use_confidence_gating: bool = True  # Gate output by confidence


class IterativeCompetition(nn.Module):
    """
    Multi-round competition with ignition dynamics.
    
    Implements the Selection phase of GWT with:
    - Iterative refinement of salience scores
    - Global ignition when activity exceeds threshold
    - Winner-take-most dynamics with soft gating
    
    Based on 2025 research on neural global workspace implementations.
    """
    
    def __init__(
        self,
        workspace_dim: int = 1024,
        num_heads: int = 16,
        selection_rounds: int = 3,
        ignition_threshold: float = 0.3,
        temperature: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.workspace_dim = workspace_dim
        self.num_heads = num_heads
        self.selection_rounds = selection_rounds
        self.ignition_threshold = ignition_threshold
        self.temperature = temperature
        
        # Self-attention for competition
        self.attention = nn.MultiheadAttention(
            embed_dim=workspace_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Recurrent refinement
        self.refine_gate = nn.Sequential(
            nn.Linear(workspace_dim * 2, workspace_dim),
            nn.Sigmoid(),
        )
        self.refine_update = nn.Sequential(
            nn.Linear(workspace_dim * 2, workspace_dim),
            nn.Tanh(),
        )
        
        # Salience accumulator
        self.salience_update = nn.GRUCell(workspace_dim, workspace_dim // 4)
        self.salience_out = nn.Linear(workspace_dim // 4, 1)
        
        # Ignition detector
        self.ignition_detector = nn.Sequential(
            nn.Linear(workspace_dim, workspace_dim // 4),
            nn.ReLU(),
            nn.Linear(workspace_dim // 4, 1),
            nn.Sigmoid(),
        )
        
        self.norm = nn.LayerNorm(workspace_dim)
    
    def forward(
        self,
        features: torch.Tensor,
        initial_saliences: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Iterative competition with ignition.
        
        Args:
            features: (batch, num_items, workspace_dim)
            initial_saliences: (batch, num_items, 1)
            
        Returns:
            winners: Selected features
            attention_weights: Final attention weights
            info: Competition metrics
        """
        batch_size, num_items, _ = features.shape
        device = features.device
        
        # Initialize salience hidden state
        salience_h = torch.zeros(
            batch_size * num_items,
            self.workspace_dim // 4,
            device=device
        )
        
        current_features = features
        saliences = initial_saliences.squeeze(-1)  # (batch, num_items)
        
        history = []
        ignition_history = []
        
        for round_idx in range(self.selection_rounds):
            # Self-attention among competitors
            attended, attn_weights = self.attention(
                current_features, current_features, current_features,
                need_weights=True,
            )
            
            # Update features with gated refinement (GRU-like)
            combined = torch.cat([current_features, attended], dim=-1)
            gate = self.refine_gate(combined)
            update = self.refine_update(combined)
            current_features = gate * current_features + (1 - gate) * update
            
            # Update saliences based on attention received
            attention_received = attn_weights.sum(dim=1)  # How much attention each item got
            
            # GRU update for salience
            flat_features = current_features.view(-1, self.workspace_dim)
            salience_h = self.salience_update(flat_features, salience_h)
            new_saliences = self.salience_out(salience_h).view(batch_size, num_items)
            
            saliences = 0.5 * saliences + 0.5 * new_saliences
            
            # Check for ignition
            ignition = self.ignition_detector(current_features.mean(dim=1))
            ignition_history.append(ignition)
            
            # Record history
            history.append({
                'saliences': saliences.clone(),
                'ignition': ignition.clone(),
            })
            
            # Early stopping on strong ignition
            if ignition.mean() > self.ignition_threshold * 1.5:
                break
        
        # Final attention weights
        attention_weights = F.softmax(saliences / self.temperature, dim=-1)
        
        # Weight features
        winners = current_features * attention_weights.unsqueeze(-1)
        winners = self.norm(winners)
        
        # Check if global ignition occurred
        final_ignition = ignition_history[-1]
        global_ignition = (final_ignition > self.ignition_threshold).float()
        
        info = {
            'ignition': final_ignition,
            'global_ignition': global_ignition,
            'selection_rounds': round_idx + 1,
            'history': history,
        }
        
        return winners, attention_weights, info


class RefinedBroadcast(nn.Module):
    """
    Iterative broadcast with feedback integration.
    
    The broadcast phase sends winning content to all specialists,
    but also receives feedback to refine the broadcast.
    
    This creates a two-way communication where:
    1. Workspace content is projected to each modality
    2. Specialists provide feedback on relevance
    3. Broadcast is refined based on feedback
    """
    
    def __init__(
        self,
        workspace_dim: int,
        modality_dims: Dict[str, int],
        broadcast_iterations: int = 2,
        broadcast_decay: float = 0.9,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.workspace_dim = workspace_dim
        self.modality_dims = modality_dims
        self.broadcast_iterations = broadcast_iterations
        self.broadcast_decay = broadcast_decay
        
        # Forward projections (workspace -> modality)
        self.forward_projections = nn.ModuleDict()
        for name, dim in modality_dims.items():
            self.forward_projections[name] = nn.Sequential(
                nn.Linear(workspace_dim, workspace_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(workspace_dim, dim),
            )
        
        # Feedback projections (modality -> workspace)
        self.feedback_projections = nn.ModuleDict()
        for name, dim in modality_dims.items():
            self.feedback_projections[name] = nn.Sequential(
                nn.Linear(dim, workspace_dim // 2),
                nn.ReLU(),
                nn.Linear(workspace_dim // 2, workspace_dim),
            )
        
        # Feedback integration
        self.feedback_gate = nn.Sequential(
            nn.Linear(workspace_dim * 2, workspace_dim),
            nn.Sigmoid(),
        )
        
        self.norm = nn.LayerNorm(workspace_dim)
    
    def forward(
        self,
        workspace_content: torch.Tensor,
        modality_states: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Iterative broadcast with feedback.
        
        Args:
            workspace_content: (batch, workspace_dim)
            modality_states: Optional current states of specialists
            
        Returns:
            broadcasts: Dict of signals per modality
            refined_content: Workspace content after feedback
        """
        current_content = workspace_content
        
        for iteration in range(self.broadcast_iterations):
            # Forward broadcast
            broadcasts = {}
            for name, projection in self.forward_projections.items():
                broadcasts[name] = projection(current_content)
            
            # Collect feedback
            feedback = torch.zeros_like(current_content)
            num_feedbacks = 0
            
            if modality_states is not None:
                for name, state in modality_states.items():
                    if name in self.feedback_projections:
                        fb = self.feedback_projections[name](state)
                        feedback = feedback + fb
                        num_feedbacks += 1
            
            if num_feedbacks > 0:
                feedback = feedback / num_feedbacks
                
                # Gate the feedback
                combined = torch.cat([current_content, feedback], dim=-1)
                gate = self.feedback_gate(combined)
                
                # Integrate with decay
                current_content = gate * current_content + (1 - gate) * feedback
                current_content = self.norm(current_content)
                
                # Apply temporal decay
                current_content = current_content * self.broadcast_decay + \
                                  workspace_content * (1 - self.broadcast_decay)
        
        return broadcasts, current_content


class SelectionBroadcastWorkspace(nn.Module):
    """
    Improved Global Workspace with explicit Selection-Broadcast cycle.
    
    Based on 2025 research on neural implementations of Global Workspace Theory.
    
    Key improvements over base GlobalWorkspace:
    1. Iterative selection with ignition dynamics
    2. Bidirectional broadcast with feedback
    3. Confidence-gated output
    4. Larger default dimensions (1024)
    
    The Selection-Broadcast cycle:
    1. SELECTION: Specialists compete, activity builds over rounds
    2. IGNITION: When activity exceeds threshold, global ignition occurs
    3. BROADCAST: Winning content is sent to all specialists
    4. FEEDBACK: Specialists respond, refining the broadcast
    """
    
    def __init__(
        self,
        config: Optional[SelectionBroadcastConfig] = None,
        modality_dims: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        super().__init__()
        
        self.config = config or SelectionBroadcastConfig(**kwargs)
        
        # Default modality dimensions (larger for better representation)
        self.modality_dims = modality_dims or {
            'vision': 1024,
            'text': 1024,
            'audio': 512,
            'sensors': 512,
        }
        
        # Modality projections
        self.projections = nn.ModuleDict()
        for name, dim in self.modality_dims.items():
            self.projections[name] = ModalityProjection(
                input_dim=dim,
                workspace_dim=self.config.workspace_dim,
                dropout=self.config.dropout,
            )
        
        # Iterative competition
        self.competition = IterativeCompetition(
            workspace_dim=self.config.workspace_dim,
            num_heads=self.config.num_heads,
            selection_rounds=self.config.selection_rounds,
            ignition_threshold=self.config.ignition_threshold,
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
        
        # Refined broadcast
        self.broadcast = RefinedBroadcast(
            workspace_dim=self.config.workspace_dim,
            modality_dims=self.modality_dims,
            broadcast_iterations=self.config.broadcast_iterations,
            broadcast_decay=self.config.broadcast_decay,
            dropout=self.config.dropout,
        )
        
        # Context integration
        self.integration = nn.Sequential(
            nn.Linear(self.config.workspace_dim * 2, self.config.workspace_dim),
            nn.LayerNorm(self.config.workspace_dim),
            nn.GELU(),  # GELU often better than ReLU for transformers
            nn.Linear(self.config.workspace_dim, self.config.workspace_dim),
        )
        
        # Confidence estimation
        if self.config.use_confidence_gating:
            self.confidence_estimator = nn.Sequential(
                nn.Linear(self.config.workspace_dim, self.config.workspace_dim // 4),
                nn.ReLU(),
                nn.Linear(self.config.workspace_dim // 4, 1),
                nn.Sigmoid(),
            )
        
        # Previous context
        self.register_buffer('prev_context', None)
    
    def reset_state(self):
        """Reset workspace state."""
        self.working_memory.reset_state()
        self.prev_context = None
    
    def forward(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        modality_states: Optional[Dict[str, torch.Tensor]] = None,
        return_details: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Process through Selection-Broadcast cycle.
        
        Args:
            modality_inputs: Dict of modality features
            modality_states: Optional specialist states for feedback
            return_details: Return detailed competition info
            
        Returns:
            Dict with workspace, broadcasts, and optional details
        """
        batch_size = next(iter(modality_inputs.values())).shape[0]
        device = next(iter(modality_inputs.values())).device
        
        # 1. Project modalities to workspace dimension
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
            raise ValueError("No valid modality inputs")
        
        projected = torch.stack(projected, dim=1)
        saliences = torch.stack(saliences, dim=1)
        
        # 2. SELECTION: Iterative competition
        winners, attention, competition_info = self.competition(projected, saliences)
        
        # 3. Aggregate to workspace content
        workspace_content = winners.sum(dim=1)
        
        # 4. Integrate with previous context
        if self.prev_context is not None:
            if self.prev_context.shape[0] != workspace_content.shape[0]:
                self.prev_context = None
        
        if self.prev_context is not None:
            combined = torch.cat([workspace_content, self.prev_context], dim=-1)
            integrated = self.integration(combined)
        else:
            integrated = workspace_content
        
        # 5. Working memory update
        memory_result = self.working_memory(integrated)
        memory_output = memory_result['output']
        
        # 6. BROADCAST: Send to specialists with feedback
        broadcasts, refined_content = self.broadcast(
            memory_output,
            modality_states=modality_states,
        )
        
        # 7. Confidence gating
        if self.config.use_confidence_gating:
            confidence = self.confidence_estimator(refined_content)
            output_workspace = refined_content * confidence
        else:
            confidence = None
            output_workspace = refined_content
        
        # 8. Update context
        self.prev_context = memory_output.detach()
        
        # Build output
        output = {
            'workspace': output_workspace,
            'broadcasts': broadcasts,
            'memory_output': memory_result,
            'modality_names': modality_names,
            'attention': {name: attention[:, i] for i, name in enumerate(modality_names)},
            'ignition': competition_info['ignition'],
            'global_ignition': competition_info['global_ignition'],
        }
        
        if confidence is not None:
            output['confidence'] = confidence
        
        if return_details:
            output['competition_details'] = competition_info
        
        return output
    
    def get_workspace_state(self) -> Dict[str, torch.Tensor]:
        """Get current workspace state."""
        return {
            'prev_context': self.prev_context,
            'memory_buffer': self.working_memory.memory_buffer,
        }


def create_selection_broadcast_workspace(
    workspace_dim: int = 1024,
    modality_dims: Optional[Dict[str, int]] = None,
    num_heads: int = 16,
    selection_rounds: int = 3,
    ignition_threshold: float = 0.3,
    memory_mode: str = "cfc",
    **kwargs,
) -> SelectionBroadcastWorkspace:
    """
    Create improved Selection-Broadcast workspace.
    
    This is the recommended workspace for new projects.
    """
    config = SelectionBroadcastConfig(
        workspace_dim=workspace_dim,
        num_heads=num_heads,
        selection_rounds=selection_rounds,
        ignition_threshold=ignition_threshold,
        memory_mode=memory_mode,
        **kwargs,
    )
    
    return SelectionBroadcastWorkspace(
        config=config,
        modality_dims=modality_dims,
    )
