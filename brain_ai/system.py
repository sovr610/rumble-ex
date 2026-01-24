"""
Brain-Inspired AI System

Complete integrated system combining all components:
- SNN Core: Spike-based feature extraction
- Modality Encoders: Vision, Text, Audio, Sensors
- HTM: Temporal sequence learning
- Global Workspace: Multi-modal integration
- Decision System: Active inference action selection
- Symbolic Reasoning: System 2 deliberation
- Meta-Learning: Adaptive plasticity control

Usage:
    from brain_ai.system import BrainAI, create_brain_ai

    # Create system
    brain = create_brain_ai(
        modalities=['vision', 'text'],
        output_type='classify',
        num_classes=10,
    )

    # Forward pass
    output = brain({
        'vision': images,
        'text': text_embeddings,
    })
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple, Union
from dataclasses import dataclass

from .config import BrainAIConfig
from .core.snn import SNNCore, ConvSNN
from .encoders.vision import VisionEncoder, create_vision_encoder
from .encoders.text import TextEncoder, create_text_encoder
from .encoders.audio import AudioEncoder, create_audio_encoder
from .encoders.sensors import SensorEncoder, create_sensor_encoder
from .encoders.engram_encoder import EngramTextEncoder, create_engram_encoder
from .temporal.htm import HTMLayer, create_htm_layer
from .workspace.global_workspace import GlobalWorkspace, create_global_workspace
from .decision.active_inference import ActiveInferenceAgent, create_active_inference_agent
from .decision.output_heads import DecisionHeads, create_decision_heads
from .reasoning.system2 import DualProcessReasoner, create_dual_process_reasoner
from .meta.neuromodulation import NeuromodulatoryGate, create_neuromodulatory_gate


@dataclass
class SystemOutput:
    """Output from the Brain-Inspired AI system."""
    output: torch.Tensor  # Main output (class logits, actions, etc.)
    workspace: torch.Tensor  # Workspace representation
    confidence: torch.Tensor  # System confidence
    attention: Optional[Dict[str, torch.Tensor]] = None  # Modality attention
    reasoning_trace: Optional[torch.Tensor] = None  # Reasoning steps
    modulators: Optional[Dict[str, torch.Tensor]] = None  # Neuromodulator states


class BrainAI(nn.Module):
    """
    Complete Brain-Inspired AI System.

    Integrates all components into a unified architecture that processes
    multi-modal inputs through brain-inspired mechanisms.

    Args:
        config: BrainAIConfig with all hyperparameters
        modalities: List of modalities to enable
        output_type: 'classify', 'generate', or 'control'
    """

    def __init__(
        self,
        config: Optional[BrainAIConfig] = None,
        modalities: Optional[List[str]] = None,
        output_type: str = "classify",
    ):
        super().__init__()

        self.config = config or BrainAIConfig()
        self.modalities = modalities or self.config.modalities
        self.output_type = output_type

        # Build components
        self._build_encoders()
        self._build_temporal()
        self._build_workspace()
        self._build_decision()
        self._build_reasoning()
        self._build_meta()

    def _build_encoders(self):
        """Build modality-specific encoders."""
        self.encoders = nn.ModuleDict()

        encoder_dim = self.config.encoder.output_dim

        if 'vision' in self.modalities:
            self.encoders['vision'] = create_vision_encoder(
                output_dim=encoder_dim,
                channels=self.config.encoder.vision_channels,
                beta=self.config.snn.beta,
                num_steps=self.config.snn.num_timesteps,
            )

        if 'text' in self.modalities:
            self.encoders['text'] = create_text_encoder(
                output_dim=encoder_dim,
                vocab_size=self.config.encoder.text_vocab_size,
                embed_dim=self.config.encoder.text_embed_dim,
                num_layers=self.config.encoder.text_num_layers,
                num_heads=self.config.encoder.text_num_heads,
            )

        if 'audio' in self.modalities:
            self.encoders['audio'] = create_audio_encoder(
                output_dim=encoder_dim,
                n_mels=self.config.encoder.audio_n_mels,
                sample_rate=self.config.encoder.audio_sample_rate,
            )

        if 'sensors' in self.modalities:
            self.encoders['sensors'] = create_sensor_encoder(
                output_dim=encoder_dim,
                hidden_dim=self.config.encoder.sensor_hidden_dim,
            )

        # Engram encoder (Phase 1 integration)
        if self.config.use_engram:
            self.encoders['engram'] = create_engram_encoder(
                output_dim=encoder_dim,
                vocab_size=self.config.engram.vocab_size,
                embedding_dim=self.config.engram.embedding_dim,
                ngram_orders=self.config.engram.ngram_orders,
                num_heads=self.config.engram.num_heads,
                table_size=self.config.engram.table_size,
            )

    def _build_temporal(self):
        """Build HTM temporal layer."""
        if self.config.use_htm:
            self.htm = create_htm_layer(
                input_size=self.config.encoder.output_dim,
                column_count=self.config.htm.column_count,
                cells_per_column=self.config.htm.cells_per_column,
                sparsity=self.config.htm.sparsity,
            )
        else:
            self.htm = None

    def _build_workspace(self):
        """Build global workspace."""
        if self.config.use_workspace:
            modality_dims = {
                name: self.config.encoder.output_dim
                for name in self.modalities
            }

            # Add engram to workspace if enabled
            if self.config.use_engram:
                modality_dims['engram'] = self.config.encoder.output_dim

            self.workspace = create_global_workspace(
                workspace_dim=self.config.workspace.workspace_dim,
                modality_dims=modality_dims,
                num_heads=self.config.workspace.num_heads,
                capacity_limit=self.config.workspace.capacity_limit,
                memory_mode=self.config.workspace.memory_mode,
                use_htm=self.config.use_htm,
                htm_layer=self.htm if self.config.use_htm else None,
            )
        else:
            self.workspace = None
            # Fallback: simple concatenation
            total_dim = len(self.modalities) * self.config.encoder.output_dim
            self.fallback_proj = nn.Linear(
                total_dim,
                self.config.workspace.workspace_dim,
            )

    def _build_decision(self):
        """Build decision system."""
        workspace_dim = self.config.workspace.workspace_dim

        # Decision heads for different output types
        self.decision_heads = create_decision_heads(
            input_dim=workspace_dim,
            num_classes=self.config.decision.num_classes,
            vocab_size=self.config.decision.text_vocab_size,
            control_dim=self.config.decision.control_dim,
        )

        # Active inference agent for action selection
        self.active_inference = create_active_inference_agent(
            obs_dim=workspace_dim,
            state_dim=64,
            action_dim=self.config.decision.num_classes,
            planning_horizon=self.config.decision.planning_horizon,
            epistemic_weight=self.config.decision.epistemic_weight,
        )

    def _build_reasoning(self):
        """Build symbolic reasoning system."""
        if self.config.use_symbolic:
            self.reasoner = create_dual_process_reasoner(
                hidden_dim=self.config.workspace.workspace_dim,
                confidence_threshold=self.config.reasoning.confidence_threshold,
                max_iterations=self.config.reasoning.num_reasoning_steps,
                use_metacognition=True,
            )
        else:
            self.reasoner = None

    def _build_meta(self):
        """Build meta-learning components."""
        if self.config.use_meta:
            self.neuromodulation = create_neuromodulatory_gate(
                input_dim=self.config.workspace.workspace_dim,
                hidden_dim=128,
            )
        else:
            self.neuromodulation = None

    def reset_state(self):
        """Reset all stateful components."""
        if self.htm is not None:
            self.htm.reset()
        if self.workspace is not None:
            self.workspace.reset_state()

    def encode(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Encode all modality inputs.

        Args:
            inputs: Dict mapping modality names to tensors

        Returns:
            Dict of encoded features (all same dimension)
        """
        encoded = {}

        for name, data in inputs.items():
            if name in self.encoders:
                encoded[name] = self.encoders[name](data)

        # Handle Engram separately - it needs token_ids
        if 'engram' in self.encoders and 'token_ids' in inputs:
            encoded['engram'] = self.encoders['engram'](inputs['token_ids'])

        return encoded

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        task: Optional[str] = None,
        return_details: bool = False,
        deterministic: bool = False,
    ) -> Union[torch.Tensor, SystemOutput]:
        """
        Full forward pass through the brain-inspired system.

        Args:
            inputs: Dict mapping modality names to input tensors
            task: Override output type ('classify', 'generate', 'control')
            return_details: Return full SystemOutput with internals
            deterministic: Use deterministic action selection

        Returns:
            Output tensor or SystemOutput with full details
        """
        task = task or self.output_type

        # 1. Encode all modalities
        encoded = self.encode(inputs)

        # 2. Global workspace integration
        if self.workspace is not None:
            ws_output = self.workspace(encoded, return_attention=True)
            workspace = ws_output['workspace']
            attention = ws_output.get('attention')
        else:
            # Fallback: concatenate and project
            features = torch.cat(list(encoded.values()), dim=-1)
            workspace = self.fallback_proj(features)
            attention = None

        # 3. Optional HTM processing
        anomaly_score = None
        if self.htm is not None and 'htm' not in (ws_output if self.workspace else {}):
            htm_out = self.htm(workspace)
            anomaly_score = htm_out.get('anomaly_likelihood')

        # 4. Symbolic reasoning (if confidence is low)
        reasoning_trace = None
        if self.reasoner is not None:
            reason_out = self.reasoner(
                workspace,
                return_details=return_details,
            )
            workspace = reason_out['output']
            confidence = reason_out['confidence']
            if return_details and 'trace' in reason_out:
                reasoning_trace = reason_out.get('trace')
        else:
            # Simple confidence from output entropy
            confidence = None

        # 5. Meta-learning modulation
        modulators = None
        if self.neuromodulation is not None:
            mod_out = self.neuromodulation(
                workspace,
                anomaly_score=anomaly_score,
                confidence=confidence,
            )
            modulators = mod_out['modulators']
            # Could modulate learning here if training

        # 6. Decision/output
        if task == 'classify':
            output_dict = self.decision_heads.classify(workspace)
            output = output_dict['logits']
            if confidence is None:
                confidence = output_dict.get('confidence')

        elif task == 'generate':
            # For generation, return workspace for decoder
            output = workspace

        elif task == 'control':
            output_dict = self.decision_heads.control_action(
                workspace,
                deterministic=deterministic,
            )
            output = output_dict['action']
            if confidence is None:
                confidence = torch.ones(output.shape[0], 1, device=output.device)

        elif task == 'active_inference':
            action, info = self.active_inference(
                workspace,
                deterministic=deterministic,
            )
            output = action
            if confidence is None:
                confidence = info['action_probs'].max(dim=-1)[0].unsqueeze(-1)

        else:
            raise ValueError(f"Unknown task: {task}")

        if return_details:
            return SystemOutput(
                output=output,
                workspace=workspace,
                confidence=confidence if confidence is not None else torch.ones_like(output[:, :1]),
                attention=attention,
                reasoning_trace=reasoning_trace,
                modulators=modulators,
            )

        return output

    def classify(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Convenience method for classification."""
        return self.forward(inputs, task='classify')

    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        max_length: int = 100,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text from inputs.

        Args:
            inputs: Multi-modal inputs
            max_length: Maximum generation length

        Returns:
            Generated token ids
        """
        workspace = self.forward(inputs, task='generate')
        return self.decision_heads.generate_text(
            workspace,
            max_length=max_length,
            **kwargs,
        )

    def act(
        self,
        inputs: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get continuous control action."""
        return self.forward(
            inputs,
            task='control',
            deterministic=deterministic,
        )


def create_brain_ai(
    modalities: List[str] = ['vision'],
    output_type: str = 'classify',
    num_classes: int = 10,
    control_dim: int = 6,
    use_htm: bool = True,
    use_symbolic: bool = True,
    use_meta: bool = True,
    use_engram: bool = False,
    workspace_dim: int = 512,
    device: str = 'auto',
    **kwargs,
) -> BrainAI:
    """
    Factory function to create Brain-Inspired AI system.

    Args:
        modalities: List of input modalities ('vision', 'text', 'audio', 'sensors')
        output_type: 'classify', 'generate', or 'control'
        num_classes: Number of classes for classification
        control_dim: Dimension of control output
        use_htm: Enable HTM temporal layer
        use_symbolic: Enable symbolic reasoning
        use_meta: Enable meta-learning modulation
        use_engram: Enable Engram text encoder for workspace competition
        workspace_dim: Dimension of global workspace
        device: Device to place model on ('auto', 'cuda', 'cpu')

    Returns:
        Configured BrainAI system
    """
    # Create config
    config = BrainAIConfig()
    config.modalities = modalities
    config.use_htm = use_htm
    config.use_symbolic = use_symbolic
    config.use_meta = use_meta
    config.use_engram = use_engram
    config.workspace.workspace_dim = workspace_dim
    config.decision.num_classes = num_classes
    config.decision.control_dim = control_dim

    # Apply any additional kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Create model
    model = BrainAI(
        config=config,
        modalities=modalities,
        output_type=output_type,
    )

    # Move to device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    return model


# Quick presets
def create_vision_classifier(
    num_classes: int = 10,
    **kwargs,
) -> BrainAI:
    """Create vision-only classifier."""
    return create_brain_ai(
        modalities=['vision'],
        output_type='classify',
        num_classes=num_classes,
        **kwargs,
    )


def create_multimodal_system(
    modalities: List[str] = ['vision', 'text'],
    **kwargs,
) -> BrainAI:
    """Create multi-modal reasoning system."""
    return create_brain_ai(
        modalities=modalities,
        output_type='classify',
        use_symbolic=True,
        **kwargs,
    )


def create_control_agent(
    modalities: List[str] = ['sensors'],
    control_dim: int = 6,
    **kwargs,
) -> BrainAI:
    """Create continuous control agent."""
    return create_brain_ai(
        modalities=modalities,
        output_type='control',
        control_dim=control_dim,
        use_meta=True,
        **kwargs,
    )
