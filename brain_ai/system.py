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
from typing import Any, Callable, Dict, Optional, List, Tuple, Union
from dataclasses import dataclass, field

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


@dataclass
class PipelineStage:
    """A single composable step in the BrainAI pipeline.

    Each stage wraps a callable that reads from and writes to a shared
    context dictionary.  Stages can be individually toggled, reordered,
    or extended without touching the ``forward()`` method.

    Attributes:
        name: Unique identifier for this stage (e.g. "encode", "workspace").
        fn: Callable that receives the context dict and returns it (mutated).
        enabled: Whether the stage will execute during ``PipelinePlan.run()``.
            Disabled stages are silently skipped.
        requires: Optional list of stage names that must appear *before*
            this stage in the plan.  Used for validation only; execution
            order is determined by the list order in ``PipelinePlan.stages``.
    """
    name: str
    fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    enabled: bool = True
    requires: Optional[List[str]] = None


class PipelinePlan:
    """Ordered, composable execution plan for the BrainAI forward pass.

    The plan is built once during ``BrainAI.__init__`` and executed on
    every ``forward()`` call.  Each stage reads from / writes to a shared
    *context* dictionary so that stages are loosely coupled and can be
    toggled or reordered freely.

    Example::

        plan = PipelinePlan()
        plan.add("encode", encode_fn)
        plan.add("workspace", workspace_fn, requires=["encode"])
        result = plan.run({"inputs": batch})
    """

    def __init__(self) -> None:
        self.stages: List[PipelineStage] = []

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def add(
        self,
        name: str,
        fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        enabled: bool = True,
        requires: Optional[List[str]] = None,
    ) -> None:
        """Append a stage to the plan.

        Args:
            name: Unique name for the stage.
            fn: Callable ``(ctx) -> ctx`` that performs the stage's work.
            enabled: If ``False`` the stage is skipped during ``run()``.
            requires: Names of stages that must precede this one.

        Raises:
            ValueError: If *name* duplicates an existing stage or if a
                required predecessor has not been added yet.
        """
        existing_names = {s.name for s in self.stages}
        if name in existing_names:
            raise ValueError(
                f"Duplicate pipeline stage name: '{name}'"
            )
        if requires:
            missing = [r for r in requires if r not in existing_names]
            if missing:
                raise ValueError(
                    f"Stage '{name}' requires stages {missing} which have "
                    f"not been added yet."
                )
        self.stages.append(PipelineStage(
            name=name,
            fn=fn,
            enabled=enabled,
            requires=requires,
        ))

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all enabled stages in order.

        Args:
            initial_context: Seed context dict; stages mutate and return it.

        Returns:
            The final context dict after all stages have run.
        """
        ctx = initial_context
        for stage in self.stages:
            if stage.enabled:
                ctx = stage.fn(ctx)
        return ctx

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        lines = ["PipelinePlan(["]
        for s in self.stages:
            tag = "ON " if s.enabled else "OFF"
            deps = f"  requires={s.requires}" if s.requires else ""
            lines.append(f"  [{tag}] {s.name}{deps}")
        lines.append("])")
        return "\n".join(lines)

    def stage_names(self) -> List[str]:
        """Return ordered list of all stage names."""
        return [s.name for s in self.stages]

    def enabled_stages(self) -> List[str]:
        """Return names of currently enabled stages."""
        return [s.name for s in self.stages if s.enabled]

    def disable(self, name: str) -> None:
        """Disable the stage with the given *name*.

        Raises:
            KeyError: If no stage with that name exists.
        """
        for s in self.stages:
            if s.name == name:
                s.enabled = False
                return
        raise KeyError(f"No pipeline stage named '{name}'")

    def enable(self, name: str) -> None:
        """Enable the stage with the given *name*.

        Raises:
            KeyError: If no stage with that name exists.
        """
        for s in self.stages:
            if s.name == name:
                s.enabled = True
                return
        raise KeyError(f"No pipeline stage named '{name}'")


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

        # Build composable pipeline plan from the constructed components
        self._pipeline = self._build_pipeline()

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
                input_dim=self.config.encoder.sensor_input_dim,
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
            # HTM input_size should match workspace_dim since it receives
            # the integrated workspace representation, not raw encoder output
            self.htm = create_htm_layer(
                input_size=self.config.workspace.workspace_dim,
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
            state_dim=self.config.decision.state_dim,
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
                hidden_dim=self.config.meta.neuromod_hidden_dim,
            )
        else:
            self.neuromodulation = None

    # ------------------------------------------------------------------
    # Pipeline plan construction
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> PipelinePlan:
        """Construct the composable PipelinePlan from current components.

        Each stage is a thin closure that reads from / writes to the
        shared context dict, preserving the exact same logic that was
        previously inlined in ``forward()``.
        """
        plan = PipelinePlan()

        # Stage 1: encode -- always enabled
        plan.add("encode", self._stage_encode, enabled=True)

        # Stage 2: workspace -- always enabled (has its own internal fallback)
        plan.add(
            "workspace", self._stage_workspace,
            enabled=True, requires=["encode"],
        )

        # Stage 3: htm -- only when the HTM layer was built
        plan.add(
            "htm", self._stage_htm,
            enabled=(self.htm is not None),
            requires=["workspace"],
        )

        # Stage 4: reasoning -- only when the symbolic reasoner was built
        plan.add(
            "reasoning", self._stage_reasoning,
            enabled=(self.reasoner is not None),
            requires=["workspace"],
        )

        # Stage 5: meta -- only when neuromodulation was built
        plan.add(
            "meta", self._stage_meta,
            enabled=(self.neuromodulation is not None),
            requires=["workspace"],
        )

        # Stage 6: decision -- always enabled
        plan.add(
            "decision", self._stage_decision,
            enabled=True, requires=["workspace"],
        )

        return plan

    # ------------------------------------------------------------------
    # Individual stage implementations
    # ------------------------------------------------------------------

    def _stage_encode(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Pipeline stage: encode all modality inputs."""
        ctx["encoded"] = self.encode(ctx["inputs"])
        return ctx

    def _stage_workspace(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Pipeline stage: global workspace integration (or fallback)."""
        encoded = ctx["encoded"]
        if self.workspace is not None:
            ws_output = self.workspace(encoded, return_attention=True)
            ctx["workspace"] = ws_output["workspace"]
            ctx["attention"] = ws_output.get("attention")
            ctx["_ws_output"] = ws_output
        else:
            features = torch.cat(list(encoded.values()), dim=-1)
            ctx["workspace"] = self.fallback_proj(features)
            ctx["attention"] = None
            ctx["_ws_output"] = {}
        return ctx

    def _stage_htm(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Pipeline stage: HTM temporal processing."""
        ws_output = ctx.get("_ws_output", {})
        if "htm" not in ws_output:
            htm_out = self.htm(ctx["workspace"])
            ctx["anomaly_score"] = htm_out.get("anomaly_likelihood")
        return ctx

    def _stage_reasoning(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Pipeline stage: symbolic reasoning (dual-process)."""
        return_details = ctx.get("return_details", False)
        reason_out = self.reasoner(
            ctx["workspace"],
            return_details=return_details,
        )
        ctx["workspace"] = reason_out["output"]
        ctx["confidence"] = reason_out["confidence"]
        if return_details and "trace" in reason_out:
            ctx["reasoning_trace"] = reason_out.get("trace")
        return ctx

    def _stage_meta(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Pipeline stage: neuromodulatory meta-learning."""
        mod_out = self.neuromodulation(
            ctx["workspace"],
            anomaly_score=ctx.get("anomaly_score"),
            confidence=ctx.get("confidence"),
        )
        ctx["modulators"] = mod_out["modulators"]
        return ctx

    def _stage_decision(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Pipeline stage: decision / output head selection."""
        task = ctx["task"]
        workspace = ctx["workspace"]
        deterministic = ctx.get("deterministic", False)
        confidence = ctx.get("confidence")

        if task == "classify":
            output_dict = self.decision_heads.classify(workspace)
            ctx["output"] = output_dict["logits"]
            if confidence is None:
                ctx["confidence"] = output_dict.get("confidence")

        elif task == "generate":
            ctx["output"] = workspace

        elif task == "control":
            output_dict = self.decision_heads.control_action(
                workspace,
                deterministic=deterministic,
            )
            ctx["output"] = output_dict["action"]
            if confidence is None:
                ctx["confidence"] = torch.ones(
                    output_dict["action"].shape[0], 1,
                    device=output_dict["action"].device,
                )

        elif task == "active_inference":
            action, info = self.active_inference(
                workspace,
                deterministic=deterministic,
            )
            ctx["output"] = action
            if confidence is None:
                if "action_probs" in info:
                    # Discrete agent: confidence from max action probability
                    ctx["confidence"] = (
                        info["action_probs"].max(dim=-1)[0].unsqueeze(-1)
                    )
                elif "action_std" in info:
                    # Continuous agent: confidence from inverse action std
                    ctx["confidence"] = (
                        1.0 / (1.0 + info["action_std"].mean(dim=-1, keepdim=True))
                    )
                else:
                    ctx["confidence"] = torch.ones(
                        action.shape[0], 1, device=action.device,
                    )

        else:
            raise ValueError(f"Unknown task: {task}")

        return ctx

    # ------------------------------------------------------------------
    # Pipeline introspection
    # ------------------------------------------------------------------

    def get_pipeline_plan(self) -> PipelinePlan:
        """Return the current pipeline plan for inspection.

        The returned ``PipelinePlan`` is the live object used by
        ``forward()``, so callers can toggle stages at runtime::

            brain.get_pipeline_plan().disable("htm")
        """
        return self._pipeline

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory-efficient training.

        Wraps the forward methods of the heaviest components (workspace,
        reasoner, encoders) with torch.utils.checkpoint so that intermediate
        activations are recomputed during the backward pass instead of being
        stored in memory.  This trades ~30% extra compute for significantly
        lower peak GPU memory, which is essential at 7B scale.
        """
        from torch.utils.checkpoint import checkpoint as _checkpoint
        import functools

        def _wrap_forward(module):
            """Return a new forward that runs the original under checkpointing."""
            original_forward = module.forward

            @functools.wraps(original_forward)
            def _checkpointed_forward(*args, **kwargs):
                # checkpoint does not natively support kwargs in older
                # PyTorch versions; use use_reentrant=False which is the
                # recommended path from PyTorch >= 2.1.
                def _run(*a):
                    return original_forward(*a, **kwargs)
                return _checkpoint(_run, *args, use_reentrant=False)

            module.forward = _checkpointed_forward
            module._gradient_checkpointing = True

        # Apply to the heaviest components
        for name, encoder in self.encoders.items():
            if not getattr(encoder, '_gradient_checkpointing', False):
                _wrap_forward(encoder)

        if self.workspace is not None and not getattr(self.workspace, '_gradient_checkpointing', False):
            _wrap_forward(self.workspace)

        if self.reasoner is not None and not getattr(self.reasoner, '_gradient_checkpointing', False):
            _wrap_forward(self.reasoner)

    def compile_model(self, backend: str = "inductor", **kwargs):
        """Apply torch.compile to performance-critical submodules.

        Compiles encoders and the global workspace -- the main compute
        bottlenecks -- using the specified backend.  Falls back silently on
        PyTorch versions that do not support ``torch.compile``.

        Args:
            backend: Compilation backend (default ``"inductor"``).
            **kwargs: Extra keyword arguments forwarded to ``torch.compile``.
        """
        if not hasattr(torch, 'compile'):
            import warnings
            warnings.warn(
                "torch.compile is not available in this PyTorch version. "
                "Skipping model compilation."
            )
            return

        for name, encoder in self.encoders.items():
            self.encoders[name] = torch.compile(encoder, backend=backend, **kwargs)

        if self.workspace is not None:
            self.workspace = torch.compile(self.workspace, backend=backend, **kwargs)

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

        Internally delegates to the composable ``PipelinePlan`` built during
        ``__init__``.  The plan executes each enabled stage in order,
        threading a shared context dict through all of them.

        Args:
            inputs: Dict mapping modality names to input tensors
            task: Override output type ('classify', 'generate', 'control')
            return_details: Return full SystemOutput with internals
            deterministic: Use deterministic action selection

        Returns:
            Output tensor or SystemOutput with full details
        """
        task = task or self.output_type

        # Seed the pipeline context with all forward() arguments
        ctx: Dict[str, Any] = {
            "inputs": inputs,
            "task": task,
            "return_details": return_details,
            "deterministic": deterministic,
            # Defaults for optional stage outputs
            "anomaly_score": None,
            "confidence": None,
            "reasoning_trace": None,
            "modulators": None,
            "attention": None,
        }

        # Run the composable pipeline
        ctx = self._pipeline.run(ctx)

        # Assemble the return value (unchanged contract)
        output = ctx["output"]

        if return_details:
            confidence = ctx.get("confidence")
            return SystemOutput(
                output=output,
                workspace=ctx["workspace"],
                confidence=confidence if confidence is not None else torch.ones_like(output[:, :1]),
                attention=ctx.get("attention"),
                reasoning_trace=ctx.get("reasoning_trace"),
                modulators=ctx.get("modulators"),
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
        # Handle nested config attributes
        elif key == 'sensor_input_dim':
            config.encoder.sensor_input_dim = value
        elif key == 'hidden_dim':
            config.snn.hidden_sizes = [value, value // 2]
        elif key == 'snn_steps':
            config.snn.num_timesteps = value
        elif key == 'htm_columns':
            config.htm.column_count = value

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
    state_dim: int = 32,
    action_dim: int = 6,
    modalities: List[str] = ['sensors'],
    control_dim: Optional[int] = None,
    **kwargs,
) -> BrainAI:
    """Create continuous control agent.
    
    Args:
        state_dim: Dimension of state/observation input
        action_dim: Dimension of action output
        modalities: Input modalities (default: ['sensors'])
        control_dim: Alias for action_dim (deprecated, use action_dim)
        **kwargs: Additional arguments passed to create_brain_ai
    """
    # Handle backwards compatibility
    if control_dim is not None:
        action_dim = control_dim
    
    # Set sensor input dim to match state_dim
    if 'sensor_input_dim' not in kwargs:
        kwargs['sensor_input_dim'] = state_dim
    
    return create_brain_ai(
        modalities=modalities,
        output_type='control',
        control_dim=action_dim,
        use_meta=True,
        **kwargs,
    )
