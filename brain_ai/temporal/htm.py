"""
Hierarchical Temporal Memory (HTM)

Implements online sequence learning with:
- Sparse Distributed Representations (SDRs)
- Spatial Pooler for input encoding
- Temporal Memory for sequence prediction
- Anomaly detection via prediction failure

Supports both htm.core library (if available) and pure PyTorch fallback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass


@dataclass
class HTMConfig:
    """Configuration for HTM layer."""
    input_size: int = 512
    column_count: int = 2048
    cells_per_column: int = 32
    sparsity: float = 0.02
    permanence_inc: float = 0.1
    permanence_dec: float = 0.1
    permanence_connected: float = 0.5
    activation_threshold: int = 13
    min_threshold: int = 10
    max_new_synapse_count: int = 20
    initial_permanence: float = 0.21


# Try to import htm.core
try:
    from htm.bindings.sdr import SDR
    from htm.bindings.algorithms import SpatialPooler as HTMSpatialPooler
    from htm.bindings.algorithms import TemporalMemory as HTMTemporalMemory
    HTM_CORE_AVAILABLE = True
except ImportError:
    HTM_CORE_AVAILABLE = False


class SparseTensor:
    """
    Sparse Distributed Representation as PyTorch tensor.

    Efficiently represents binary sparse vectors where only ~2% of bits are active.
    """

    def __init__(self, size: int, device: torch.device = None):
        self.size = size
        self.device = device or torch.device('cpu')
        self._sparse_indices: Optional[torch.Tensor] = None

    @property
    def sparse(self) -> torch.Tensor:
        """Get active indices."""
        return self._sparse_indices if self._sparse_indices is not None else torch.tensor([], device=self.device)

    @sparse.setter
    def sparse(self, indices: Union[torch.Tensor, List[int], np.ndarray]):
        """Set active indices."""
        if isinstance(indices, np.ndarray):
            indices = torch.from_numpy(indices)
        elif isinstance(indices, list):
            indices = torch.tensor(indices)
        self._sparse_indices = indices.to(self.device).long()

    def to_dense(self) -> torch.Tensor:
        """Convert to dense binary tensor."""
        dense = torch.zeros(self.size, device=self.device)
        if self._sparse_indices is not None and len(self._sparse_indices) > 0:
            dense[self._sparse_indices] = 1.0
        return dense

    @classmethod
    def from_dense(cls, dense: torch.Tensor, threshold: float = 0.5) -> 'SparseTensor':
        """Create from dense tensor via thresholding."""
        sdr = cls(dense.shape[-1], dense.device)
        sdr.sparse = torch.where(dense > threshold)[0]
        return sdr

    def overlap(self, other: 'SparseTensor') -> int:
        """Count overlapping active bits."""
        if self._sparse_indices is None or other._sparse_indices is None:
            return 0
        set1 = set(self._sparse_indices.tolist())
        set2 = set(other._sparse_indices.tolist())
        return len(set1 & set2)


class PytorchSpatialPooler(nn.Module):
    """
    Spatial Pooler in pure PyTorch.

    Converts input patterns to sparse distributed representations (SDRs).
    Learns to recognize common patterns and maintain consistent sparsity.
    """

    def __init__(
        self,
        input_size: int,
        column_count: int = 2048,
        sparsity: float = 0.02,
        potential_radius: Optional[int] = None,
        potential_pct: float = 0.85,
        permanence_inc: float = 0.1,
        permanence_dec: float = 0.1,
        permanence_connected: float = 0.5,
        boost_strength: float = 3.0,
    ):
        super().__init__()

        self.input_size = input_size
        self.column_count = column_count
        self.sparsity = sparsity
        self.num_active = int(column_count * sparsity)
        self.potential_radius = potential_radius or input_size
        self.potential_pct = potential_pct
        self.permanence_inc = permanence_inc
        self.permanence_dec = permanence_dec
        self.permanence_connected = permanence_connected
        self.boost_strength = boost_strength

        # Permanence values (learnable connections)
        # Initialize with random values around connected threshold
        permanences = torch.rand(column_count, input_size) * 0.2 + 0.1
        self.register_buffer('permanences', permanences)

        # Potential synapses mask (which inputs each column can connect to)
        potential_mask = torch.rand(column_count, input_size) < potential_pct
        self.register_buffer('potential_mask', potential_mask.float())

        # Boosting factors
        self.register_buffer('boost_factors', torch.ones(column_count))
        self.register_buffer('active_duty_cycles', torch.zeros(column_count))
        self.register_buffer('iteration_count', torch.tensor(0))

    def compute_overlap(self, input_sdr: torch.Tensor) -> torch.Tensor:
        """Compute overlap scores between input and columns."""
        # Connected synapses
        connected = (self.permanences >= self.permanence_connected).float()
        connected = connected * self.potential_mask

        # Overlap = sum of connected synapses to active inputs
        overlap = torch.matmul(connected, input_sdr)

        # Apply boosting
        overlap = overlap * self.boost_factors

        return overlap

    def inhibit(self, overlap: torch.Tensor) -> torch.Tensor:
        """Global inhibition: select top-k columns."""
        # Get indices of top-k overlaps
        _, top_indices = torch.topk(overlap, self.num_active)

        # Create binary active columns tensor
        active = torch.zeros_like(overlap)
        active[top_indices] = 1.0

        return active

    def learn(self, input_sdr: torch.Tensor, active_columns: torch.Tensor):
        """Update permanences based on Hebbian learning."""
        # Only learn for active columns
        active_mask = active_columns.unsqueeze(-1)  # (columns, 1)
        input_mask = input_sdr.unsqueeze(0)  # (1, inputs)

        # Increase permanence for active inputs to active columns
        delta_pos = self.permanence_inc * active_mask * input_mask * self.potential_mask

        # Decrease permanence for inactive inputs to active columns
        delta_neg = self.permanence_dec * active_mask * (1 - input_mask) * self.potential_mask

        # Update permanences
        self.permanences = torch.clamp(
            self.permanences + delta_pos - delta_neg,
            0.0, 1.0
        )

        # Update duty cycles and boosting
        self._update_boosting(active_columns)

    def _update_boosting(self, active_columns: torch.Tensor):
        """Update boosting factors based on activity."""
        self.iteration_count += 1
        period = min(1000, self.iteration_count.item())

        # Exponential moving average of activity
        alpha = 1.0 / period
        self.active_duty_cycles = (1 - alpha) * self.active_duty_cycles + alpha * active_columns

        # Target duty cycle
        target_duty = self.sparsity

        # Boost factors: increase for underactive columns
        self.boost_factors = torch.exp(
            self.boost_strength * (target_duty - self.active_duty_cycles)
        )

    def forward(
        self,
        input_sdr: torch.Tensor,
        learn: bool = True
    ) -> torch.Tensor:
        """
        Process input through spatial pooler.

        Args:
            input_sdr: Binary input (input_size,) or (batch, input_size)
            learn: Whether to update permanences

        Returns:
            active_columns: Binary output (column_count,) or (batch, column_count)
        """
        # Handle batched input
        if input_sdr.dim() == 2:
            # Process batch sequentially (HTM is typically online)
            outputs = []
            for i in range(input_sdr.shape[0]):
                out = self._forward_single(input_sdr[i], learn)
                outputs.append(out)
            return torch.stack(outputs)

        return self._forward_single(input_sdr, learn)

    def _forward_single(self, input_sdr: torch.Tensor, learn: bool) -> torch.Tensor:
        """Process single input."""
        overlap = self.compute_overlap(input_sdr)
        active_columns = self.inhibit(overlap)

        if learn:
            self.learn(input_sdr, active_columns)

        return active_columns


class PytorchTemporalMemory(nn.Module):
    """
    Temporal Memory in pure PyTorch.

    Learns sequences by forming connections between cells in different columns.
    Cells in a column represent different contexts for the same input pattern.
    """

    def __init__(
        self,
        column_count: int = 2048,
        cells_per_column: int = 32,
        activation_threshold: int = 13,
        min_threshold: int = 10,
        max_new_synapse_count: int = 20,
        initial_permanence: float = 0.21,
        permanence_connected: float = 0.5,
        permanence_inc: float = 0.1,
        permanence_dec: float = 0.1,
    ):
        super().__init__()

        self.column_count = column_count
        self.cells_per_column = cells_per_column
        self.num_cells = column_count * cells_per_column
        self.activation_threshold = activation_threshold
        self.min_threshold = min_threshold
        self.max_new_synapse_count = max_new_synapse_count
        self.initial_permanence = initial_permanence
        self.permanence_connected = permanence_connected
        self.permanence_inc = permanence_inc
        self.permanence_dec = permanence_dec

        # Segments and synapses are stored as sparse structures
        # For efficiency, we use a simplified representation
        # In production, use proper sparse data structures

        # Segment connections: learned through usage
        # Using a dense approximation for simplicity
        self.register_buffer(
            'segment_weights',
            torch.zeros(self.num_cells, self.num_cells) * 0.01
        )

        # Cell states
        self.register_buffer('active_cells', torch.zeros(self.num_cells))
        self.register_buffer('winner_cells', torch.zeros(self.num_cells))
        self.register_buffer('predictive_cells', torch.zeros(self.num_cells))

        # Previous states for learning
        self.register_buffer('prev_active_cells', torch.zeros(self.num_cells))
        self.register_buffer('prev_winner_cells', torch.zeros(self.num_cells))

    def reset(self):
        """Reset cell states for new sequence."""
        self.active_cells.zero_()
        self.winner_cells.zero_()
        self.predictive_cells.zero_()
        self.prev_active_cells.zero_()
        self.prev_winner_cells.zero_()

    def compute_activity(self, active_columns: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute active and predictive cells based on active columns.

        Returns:
            active_cells: Currently active cells
            predictive_cells: Cells predicted for next step
        """
        # Expand columns to cells
        # Each column has cells_per_column cells
        column_indices = torch.where(active_columns > 0)[0]

        # Cells that were predicted AND are in active columns
        predicted_active = torch.zeros(self.num_cells, device=active_columns.device)

        for col_idx in column_indices:
            start_cell = col_idx * self.cells_per_column
            end_cell = start_cell + self.cells_per_column

            # Check if any cell in this column was predicted
            col_predictions = self.predictive_cells[start_cell:end_cell]

            if col_predictions.sum() > 0:
                # Activate predicted cells
                predicted_active[start_cell:end_cell] = col_predictions[:]
            else:
                # Bursting: activate all cells in column
                predicted_active[start_cell:end_cell] = 1.0

        # New active cells
        new_active = predicted_active

        # Compute predictions for next step
        # Cells with enough active presynaptic connections
        connected_weights = (self.segment_weights >= self.permanence_connected).float()
        presynaptic_activity = torch.matmul(connected_weights, new_active)
        new_predictive = (presynaptic_activity >= self.activation_threshold).float()

        return new_active, new_predictive

    def learn(self, active_columns: torch.Tensor):
        """Learn from current activation pattern."""
        if self.prev_active_cells.sum() == 0:
            return  # Nothing to learn from

        # Strengthen connections from previously active cells to currently active cells
        # This is a simplified Hebbian learning rule

        active_mask = self.active_cells.unsqueeze(0)  # (1, num_cells)
        prev_active_mask = self.prev_active_cells.unsqueeze(1)  # (num_cells, 1)

        # Hebbian update: strengthen connections from prev_active to active
        delta = self.permanence_inc * active_mask * prev_active_mask

        # Decay unused connections slightly
        decay = self.permanence_dec * 0.1 * (1 - active_mask) * prev_active_mask

        self.segment_weights = torch.clamp(
            self.segment_weights + delta - decay,
            0.0, 1.0
        )

    def forward(
        self,
        active_columns: torch.Tensor,
        learn: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Process active columns through temporal memory.

        Args:
            active_columns: Binary column activations (column_count,)
            learn: Whether to update synapses

        Returns:
            Dict with active_cells, predictive_cells, anomaly
        """
        # Save previous state
        self.prev_active_cells = self.active_cells.clone()
        self.prev_winner_cells = self.winner_cells.clone()

        # Compute new activations
        self.active_cells, new_predictive = self.compute_activity(active_columns)

        # Compute anomaly: fraction of active columns that weren't predicted
        column_indices = torch.where(active_columns > 0)[0]
        num_predicted = 0

        for col_idx in column_indices:
            start_cell = col_idx * self.cells_per_column
            end_cell = start_cell + self.cells_per_column
            if self.predictive_cells[start_cell:end_cell].sum() > 0:
                num_predicted += 1

        anomaly = 1.0 - (num_predicted / max(len(column_indices), 1))

        # Update predictive cells for next step
        self.predictive_cells = new_predictive

        # Learn
        if learn:
            self.learn(active_columns)

        return {
            'active_cells': self.active_cells.clone(),
            'predictive_cells': self.predictive_cells.clone(),
            'anomaly': torch.tensor(anomaly, device=active_columns.device),
        }


class HTMLayer(nn.Module):
    """
    Complete HTM Layer combining Spatial Pooler and Temporal Memory.

    Uses htm.core if available, otherwise falls back to PyTorch implementation.

    Args:
        config: HTMConfig with all parameters
        use_htm_core: Force use of htm.core (will fail if not available)
    """

    def __init__(
        self,
        config: Optional[HTMConfig] = None,
        use_htm_core: bool = False,
    ):
        super().__init__()

        self.config = config or HTMConfig()
        self.use_htm_core = use_htm_core and HTM_CORE_AVAILABLE

        if self.use_htm_core:
            self._init_htm_core()
        else:
            self._init_pytorch()

        # Anomaly tracking for likelihood computation
        self.anomaly_history: List[float] = []
        self.anomaly_window = 1000

    def _init_htm_core(self):
        """Initialize with htm.core library."""
        cfg = self.config

        self.input_sdr = SDR(cfg.input_size)
        self.active_columns_sdr = SDR(cfg.column_count)

        self.sp = HTMSpatialPooler(
            inputDimensions=[cfg.input_size],
            columnDimensions=[cfg.column_count],
            potentialRadius=cfg.input_size,
            potentialPct=0.85,
            globalInhibition=True,
            localAreaDensity=cfg.sparsity,
            synPermInactiveDec=cfg.permanence_dec,
            synPermActiveInc=cfg.permanence_inc,
            synPermConnected=cfg.permanence_connected,
            boostStrength=3.0,
        )

        self.tm = HTMTemporalMemory(
            columnDimensions=[cfg.column_count],
            cellsPerColumn=cfg.cells_per_column,
            activationThreshold=cfg.activation_threshold,
            initialPermanence=cfg.initial_permanence,
            connectedPermanence=cfg.permanence_connected,
            minThreshold=cfg.min_threshold,
            maxNewSynapseCount=cfg.max_new_synapse_count,
            permanenceIncrement=cfg.permanence_inc,
            permanenceDecrement=cfg.permanence_dec,
        )

    def _init_pytorch(self):
        """Initialize with pure PyTorch implementation."""
        cfg = self.config

        self.sp = PytorchSpatialPooler(
            input_size=cfg.input_size,
            column_count=cfg.column_count,
            sparsity=cfg.sparsity,
            permanence_inc=cfg.permanence_inc,
            permanence_dec=cfg.permanence_dec,
            permanence_connected=cfg.permanence_connected,
        )

        self.tm = PytorchTemporalMemory(
            column_count=cfg.column_count,
            cells_per_column=cfg.cells_per_column,
            activation_threshold=cfg.activation_threshold,
            min_threshold=cfg.min_threshold,
            max_new_synapse_count=cfg.max_new_synapse_count,
            initial_permanence=cfg.initial_permanence,
            permanence_connected=cfg.permanence_connected,
            permanence_inc=cfg.permanence_inc,
            permanence_dec=cfg.permanence_dec,
        )

    def reset(self):
        """Reset temporal memory state."""
        if self.use_htm_core:
            self.tm.reset()
        else:
            self.tm.reset()
        self.anomaly_history = []

    def _compute_anomaly_likelihood(self, anomaly: float) -> float:
        """Compute smoothed anomaly likelihood."""
        self.anomaly_history.append(anomaly)
        if len(self.anomaly_history) > self.anomaly_window:
            self.anomaly_history.pop(0)

        if len(self.anomaly_history) < 10:
            return anomaly

        mean_anomaly = np.mean(self.anomaly_history)
        std_anomaly = np.std(self.anomaly_history) + 1e-6

        likelihood = 1 - np.exp(-(anomaly - mean_anomaly) / std_anomaly)
        return float(np.clip(likelihood, 0, 1))

    def forward(
        self,
        x: torch.Tensor,
        learn: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Process input through HTM.

        Args:
            x: Input features (batch, input_size) or (input_size,)
            learn: Whether to learn from input

        Returns:
            Dict with:
                - features: Processed features for downstream
                - active_cells: Currently active cells
                - predictive_cells: Predicted cells for next step
                - anomaly: Raw anomaly score
                - anomaly_likelihood: Smoothed anomaly probability
        """
        # Handle batched input
        if x.dim() == 2:
            results = []
            for i in range(x.shape[0]):
                result = self._forward_single(x[i], learn)
                results.append(result)

            # Stack results
            return {
                'features': torch.stack([r['features'] for r in results]),
                'active_cells': torch.stack([r['active_cells'] for r in results]),
                'predictive_cells': torch.stack([r['predictive_cells'] for r in results]),
                'anomaly': torch.stack([r['anomaly'] for r in results]),
                'anomaly_likelihood': torch.stack([r['anomaly_likelihood'] for r in results]),
            }

        return self._forward_single(x, learn)

    def _forward_single(
        self,
        x: torch.Tensor,
        learn: bool
    ) -> Dict[str, torch.Tensor]:
        """Process single input."""
        device = x.device

        if self.use_htm_core:
            return self._forward_htm_core(x, learn, device)
        else:
            return self._forward_pytorch(x, learn, device)

    def _forward_htm_core(
        self,
        x: torch.Tensor,
        learn: bool,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Forward pass using htm.core."""
        # Convert to SDR
        x_np = x.detach().cpu().numpy()
        self.input_sdr.sparse = np.where(x_np > 0.5)[0].tolist()

        # Spatial pooler
        self.sp.compute(self.input_sdr, learn, self.active_columns_sdr)

        # Temporal memory
        self.tm.compute(self.active_columns_sdr, learn)

        # Get results
        anomaly = float(self.tm.anomaly)
        anomaly_likelihood = self._compute_anomaly_likelihood(anomaly)

        active_cells = torch.zeros(self.config.column_count * self.config.cells_per_column, device=device)
        active_cells[self.tm.getActiveCells().sparse] = 1.0

        predictive_cells = torch.zeros(self.config.column_count * self.config.cells_per_column, device=device)
        predictive_cells[self.tm.getPredictiveCells().sparse] = 1.0

        # Features: active column representation
        features = torch.zeros(self.config.column_count, device=device)
        features[self.active_columns_sdr.sparse] = 1.0

        return {
            'features': features,
            'active_cells': active_cells,
            'predictive_cells': predictive_cells,
            'anomaly': torch.tensor(anomaly, device=device),
            'anomaly_likelihood': torch.tensor(anomaly_likelihood, device=device),
        }

    def _forward_pytorch(
        self,
        x: torch.Tensor,
        learn: bool,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Forward pass using PyTorch implementation."""
        # Binarize input
        x_binary = (x > 0.5).float()

        # Spatial pooler
        active_columns = self.sp(x_binary, learn=learn)

        # Temporal memory
        tm_result = self.tm(active_columns, learn=learn)

        anomaly = tm_result['anomaly'].item()
        anomaly_likelihood = self._compute_anomaly_likelihood(anomaly)

        return {
            'features': active_columns,
            'active_cells': tm_result['active_cells'],
            'predictive_cells': tm_result['predictive_cells'],
            'anomaly': tm_result['anomaly'],
            'anomaly_likelihood': torch.tensor(anomaly_likelihood, device=device),
        }


# Convenience function
def create_htm_layer(
    input_size: int = 512,
    column_count: int = 2048,
    cells_per_column: int = 32,
    sparsity: float = 0.02,
    **kwargs
) -> HTMLayer:
    """Create HTM layer with specified parameters."""
    config = HTMConfig(
        input_size=input_size,
        column_count=column_count,
        cells_per_column=cells_per_column,
        sparsity=sparsity,
        **kwargs
    )
    return HTMLayer(config)
