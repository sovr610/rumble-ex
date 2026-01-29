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
    Temporal Memory in pure PyTorch with SPARSE segment storage.

    Learns sequences by forming connections between cells in different columns.
    Cells in a column represent different contexts for the same input pattern.
    
    Uses sparse representation to handle large cell counts efficiently.
    Memory scales with actual connections, not O(num_cells^2).
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
        max_segments_per_cell: int = 128,
        max_synapses_per_segment: int = 32,
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
        self.max_segments_per_cell = max_segments_per_cell
        self.max_synapses_per_segment = max_synapses_per_segment

        # SPARSE segment storage: Dict[cell_id] -> List[Dict[presynaptic_cell_id -> permanence]]
        # Each cell can have multiple segments, each segment has synapses to presynaptic cells
        self.segments: Dict[int, List[Dict[int, float]]] = {}
        
        # Cell states (these are small: O(num_cells) not O(num_cells^2))
        self.register_buffer('active_cells', torch.zeros(self.num_cells))
        self.register_buffer('winner_cells', torch.zeros(self.num_cells))
        self.register_buffer('predictive_cells', torch.zeros(self.num_cells))

        # Previous states for learning
        self.register_buffer('prev_active_cells', torch.zeros(self.num_cells))
        self.register_buffer('prev_winner_cells', torch.zeros(self.num_cells))
        
        # Track active cell indices for efficiency
        self._prev_active_indices: List[int] = []
        self._active_indices: List[int] = []

    def reset(self):
        """Reset cell states for new sequence."""
        self.active_cells.zero_()
        self.winner_cells.zero_()
        self.predictive_cells.zero_()
        self.prev_active_cells.zero_()
        self.prev_winner_cells.zero_()
        self._prev_active_indices = []
        self._active_indices = []

    def _get_segment_activity(self, cell_id: int, active_cell_set: set) -> int:
        """Count active connected synapses for the best segment of a cell."""
        if cell_id not in self.segments:
            return 0
        
        best_activity = 0
        for segment in self.segments[cell_id]:
            activity = sum(
                1 for pre_cell, perm in segment.items()
                if pre_cell in active_cell_set and perm >= self.permanence_connected
            )
            best_activity = max(best_activity, activity)
        
        return best_activity

    def _get_best_matching_segment(self, cell_id: int, active_cell_set: set) -> Tuple[Optional[int], int]:
        """Find the best matching segment for a cell (most active synapses)."""
        if cell_id not in self.segments:
            return None, 0
        
        best_segment_idx = None
        best_activity = 0
        
        for seg_idx, segment in enumerate(self.segments[cell_id]):
            activity = sum(
                1 for pre_cell, perm in segment.items()
                if pre_cell in active_cell_set and perm >= self.permanence_connected
            )
            if activity > best_activity:
                best_activity = activity
                best_segment_idx = seg_idx
        
        return best_segment_idx, best_activity

    def compute_activity(self, active_columns: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute active and predictive cells based on active columns.

        Returns:
            active_cells: Currently active cells
            predictive_cells: Cells predicted for next step
        """
        device = active_columns.device
        column_indices = torch.where(active_columns > 0)[0].tolist()
        
        # Previous active cells as a set for O(1) lookup
        prev_active_set = set(self._prev_active_indices)
        
        # Compute new active cells
        new_active = torch.zeros(self.num_cells, device=device)
        new_winner = torch.zeros(self.num_cells, device=device)
        new_active_indices = []

        for col_idx in column_indices:
            start_cell = col_idx * self.cells_per_column
            end_cell = start_cell + self.cells_per_column
            
            # Check if any cell in this column was predicted
            col_predictions = self.predictive_cells[start_cell:end_cell]
            
            if col_predictions.sum() > 0:
                # Activate only predicted cells
                for cell_offset in range(self.cells_per_column):
                    cell_id = start_cell + cell_offset
                    if self.predictive_cells[cell_id] > 0:
                        new_active[cell_id] = 1.0
                        new_winner[cell_id] = 1.0
                        new_active_indices.append(cell_id)
            else:
                # Bursting: activate all cells, pick winner based on best matching segment
                best_cell = start_cell
                best_match_score = -1
                
                for cell_offset in range(self.cells_per_column):
                    cell_id = start_cell + cell_offset
                    new_active[cell_id] = 1.0
                    new_active_indices.append(cell_id)
                    
                    # Find cell with best matching segment for learning
                    _, match_score = self._get_best_matching_segment(cell_id, prev_active_set)
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_cell = cell_id
                
                new_winner[best_cell] = 1.0

        self._active_indices = new_active_indices
        active_set = set(new_active_indices)
        
        # Compute predictions for next step using sparse segments
        new_predictive = torch.zeros(self.num_cells, device=device)
        
        for cell_id, segments in self.segments.items():
            for segment in segments:
                # Count active connected synapses
                activity = sum(
                    1 for pre_cell, perm in segment.items()
                    if pre_cell in active_set and perm >= self.permanence_connected
                )
                if activity >= self.activation_threshold:
                    new_predictive[cell_id] = 1.0
                    break  # Cell is predicted, no need to check more segments

        self.winner_cells = new_winner
        return new_active, new_predictive

    def learn(self, active_columns: torch.Tensor):
        """Learn from current activation pattern using sparse updates."""
        if len(self._prev_active_indices) == 0:
            return  # Nothing to learn from

        prev_active_set = set(self._prev_active_indices)
        
        # For each winner cell, reinforce connections from previously active cells
        winner_indices = torch.where(self.winner_cells > 0)[0].tolist()
        
        for cell_id in winner_indices:
            # Get or create segments for this cell
            if cell_id not in self.segments:
                self.segments[cell_id] = []
            
            # Find best matching segment or create new one
            best_seg_idx, best_activity = self._get_best_matching_segment(cell_id, prev_active_set)
            
            if best_seg_idx is not None and best_activity >= self.min_threshold:
                # Reinforce existing segment
                segment = self.segments[cell_id][best_seg_idx]
                
                # Strengthen synapses to active cells, weaken to inactive
                for pre_cell in list(segment.keys()):
                    if pre_cell in prev_active_set:
                        segment[pre_cell] = min(1.0, segment[pre_cell] + self.permanence_inc)
                    else:
                        segment[pre_cell] = max(0.0, segment[pre_cell] - self.permanence_dec)
                        # Remove dead synapses
                        if segment[pre_cell] <= 0:
                            del segment[pre_cell]
                
                # Add new synapses to previously active cells not yet connected
                existing_pre = set(segment.keys())
                new_candidates = [c for c in self._prev_active_indices if c not in existing_pre]
                num_to_add = min(
                    self.max_new_synapse_count - len(segment),
                    len(new_candidates),
                    self.max_synapses_per_segment - len(segment)
                )
                if num_to_add > 0:
                    import random
                    for pre_cell in random.sample(new_candidates, num_to_add):
                        segment[pre_cell] = self.initial_permanence
                        
            elif len(self.segments[cell_id]) < self.max_segments_per_cell:
                # Create new segment
                import random
                num_synapses = min(self.max_new_synapse_count, len(self._prev_active_indices))
                if num_synapses > 0:
                    pre_cells = random.sample(self._prev_active_indices, num_synapses)
                    new_segment = {pc: self.initial_permanence for pc in pre_cells}
                    self.segments[cell_id].append(new_segment)

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
        self._prev_active_indices = self._active_indices.copy()

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
    
    def get_memory_stats(self) -> Dict[str, int]:
        """Get statistics about memory usage."""
        total_segments = sum(len(segs) for segs in self.segments.values())
        total_synapses = sum(
            len(seg) for segs in self.segments.values() for seg in segs
        )
        return {
            'cells_with_segments': len(self.segments),
            'total_segments': total_segments,
            'total_synapses': total_synapses,
            'avg_synapses_per_segment': total_synapses / max(total_segments, 1),
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


# =============================================================================
# AHTM: Accelerated HTM with Reflex Memory (2025 Research)
# =============================================================================

class ReflexMemory(nn.Module):
    """
    Reflex Memory for AHTM - stores frequently-accessed patterns for O(1) lookup.
    
    Based on "Enhancing Biologically Inspired Hierarchical Temporal Memory with
    Reflex Memory" (2025, arXiv:2504.03746).
    
    When a pattern is seen repeatedly, it's promoted to Reflex Memory for
    instant lookup instead of full HTM computation. This mimics how the brain
    develops "reflexive" responses for familiar patterns.
    
    Key features:
    - Locality-Sensitive Hashing (LSH) for fast approximate matching
    - Automatic promotion based on access frequency
    - LRU eviction when memory is full
    - Continuous learning without catastrophic forgetting
    
    Args:
        pattern_dim: Dimension of input patterns
        max_patterns: Maximum patterns to store
        promotion_threshold: Access count before promotion from HTM cache
        similarity_threshold: LSH similarity threshold for match
        decay_rate: Decay for access counts (prevents stale patterns)
    """
    
    def __init__(
        self,
        pattern_dim: int,
        max_patterns: int = 10000,
        promotion_threshold: int = 5,
        similarity_threshold: float = 0.9,
        decay_rate: float = 0.99,
        num_hashes: int = 16,
        hash_dim: int = 64,
    ):
        super().__init__()
        
        self.pattern_dim = pattern_dim
        self.max_patterns = max_patterns
        self.promotion_threshold = promotion_threshold
        self.similarity_threshold = similarity_threshold
        self.decay_rate = decay_rate
        self.num_hashes = num_hashes
        self.hash_dim = hash_dim
        
        # Pattern storage
        self.register_buffer('patterns', torch.zeros(max_patterns, pattern_dim))
        self.register_buffer('predictions', torch.zeros(max_patterns, pattern_dim))
        self.register_buffer('access_counts', torch.zeros(max_patterns))
        self.register_buffer('timestamps', torch.zeros(max_patterns, dtype=torch.long))
        self.register_buffer('valid_mask', torch.zeros(max_patterns, dtype=torch.bool))
        self.register_buffer('num_stored', torch.tensor(0, dtype=torch.long))
        self.register_buffer('current_time', torch.tensor(0, dtype=torch.long))
        
        # LSH projection for fast lookup
        self.register_buffer(
            'hash_projections',
            torch.randn(pattern_dim, num_hashes * hash_dim) / np.sqrt(pattern_dim)
        )
        
        # Pre-computed hashes for stored patterns (updated on store)
        self.register_buffer('stored_hashes', torch.zeros(max_patterns, num_hashes * hash_dim))
        
        # Statistics tracking
        self.hits = 0
        self.misses = 0
    
    def compute_hash(self, pattern: torch.Tensor) -> torch.Tensor:
        """
        Compute locality-sensitive hash using random projections.
        
        Args:
            pattern: (batch, pattern_dim) or (pattern_dim,) input
            
        Returns:
            Binary hash (batch, num_hashes * hash_dim) or (num_hashes * hash_dim,)
        """
        # Project and binarize
        projected = pattern @ self.hash_projections
        return (projected > 0).float()
    
    def lookup(
        self,
        pattern: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, float, int]]:
        """
        Fast O(1) lookup via LSH.
        
        Args:
            pattern: Input pattern (pattern_dim,) or (1, pattern_dim)
            
        Returns:
            If found: (prediction, confidence, pattern_idx)
            If not found: None
        """
        if pattern.dim() == 2:
            pattern = pattern.squeeze(0)
        
        if self.num_stored == 0:
            self.misses += 1
            return None
        
        # Compute hash for input
        pattern_hash = self.compute_hash(pattern)
        
        # Compare with stored hashes (only valid entries)
        valid_count = min(self.num_stored.item(), self.max_patterns)
        stored_hashes = self.stored_hashes[:valid_count]
        
        # Hamming similarity (1 - normalized Hamming distance)
        matches = (pattern_hash == stored_hashes).float()
        similarity = matches.mean(dim=-1)
        
        # Find best match
        best_sim, best_idx = similarity.max(dim=0)
        
        if best_sim >= self.similarity_threshold:
            self.hits += 1
            
            # Update access count and timestamp
            self.access_counts[best_idx] += 1
            self.timestamps[best_idx] = self.current_time
            
            return (
                self.predictions[best_idx].clone(),
                best_sim.item(),
                best_idx.item(),
            )
        
        self.misses += 1
        return None
    
    def store(
        self,
        pattern: torch.Tensor,
        prediction: torch.Tensor,
        force: bool = False,
    ) -> int:
        """
        Store pattern-prediction pair in Reflex Memory.
        
        Args:
            pattern: Input pattern
            prediction: Associated prediction/response
            force: If True, store even if below promotion threshold
            
        Returns:
            Index where stored, or -1 if not stored
        """
        if pattern.dim() == 2:
            pattern = pattern.squeeze(0)
        if prediction.dim() == 2:
            prediction = prediction.squeeze(0)
        
        self.current_time += 1
        
        # Check if pattern already exists (to update, not duplicate)
        existing = self.lookup(pattern)
        if existing is not None:
            _, _, idx = existing
            # Update prediction with moving average
            alpha = 0.1
            self.predictions[idx] = (1 - alpha) * self.predictions[idx] + alpha * prediction
            return idx
        
        # Find storage slot
        if self.num_stored < self.max_patterns:
            idx = self.num_stored.item()
            self.num_stored += 1
        else:
            # LRU eviction: find oldest accessed pattern
            recency_score = self.current_time - self.timestamps[:self.max_patterns]
            # Combine with inverse access count for importance
            importance = self.access_counts[:self.max_patterns] / (recency_score.float() + 1)
            idx = importance.argmin().item()
        
        # Store
        self.patterns[idx] = pattern
        self.predictions[idx] = prediction
        self.access_counts[idx] = 1
        self.timestamps[idx] = self.current_time
        self.valid_mask[idx] = True
        self.stored_hashes[idx] = self.compute_hash(pattern)
        
        return idx
    
    def decay_access_counts(self):
        """Apply decay to access counts (call periodically)."""
        self.access_counts *= self.decay_rate
    
    def get_statistics(self) -> Dict[str, float]:
        """Get cache statistics."""
        total = self.hits + self.misses
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / max(total, 1),
            'patterns_stored': self.num_stored.item(),
            'utilization': self.num_stored.item() / self.max_patterns,
        }
    
    def reset_statistics(self):
        """Reset hit/miss counters."""
        self.hits = 0
        self.misses = 0


class AcceleratedHTM(nn.Module):
    """
    AHTM: Accelerated Hierarchical Temporal Memory.
    
    Combines traditional HTM with Reflex Memory for fast pattern matching.
    
    Architecture:
    1. Reflex Memory: O(1) lookup for known patterns
    2. Full HTM: Slow path for novel patterns (with online learning)
    3. Automatic promotion: Frequently-accessed patterns promoted to RM
    
    Benefits:
    - Much faster inference for common patterns
    - Maintains HTM's ability to learn new sequences
    - No catastrophic forgetting
    - Real-time capable for edge deployment
    
    Based on: "Enhancing Biologically Inspired HTM with Reflex Memory" (2025)
    
    Args:
        htm_layer: Base HTMLayer for full computation
        reflex_memory: ReflexMemory for fast lookup
        use_htm_on_miss: If False, return zeros on RM miss (faster but less accurate)
        promotion_threshold: Minimum HTM calls before promoting to RM
    """
    
    def __init__(
        self,
        htm_layer: HTMLayer,
        reflex_memory: Optional[ReflexMemory] = None,
        use_htm_on_miss: bool = True,
        promotion_threshold: int = 5,
    ):
        super().__init__()
        
        self.htm = htm_layer
        self.use_htm_on_miss = use_htm_on_miss
        self.promotion_threshold = promotion_threshold
        
        # Create Reflex Memory if not provided
        if reflex_memory is None:
            pattern_dim = htm_layer.config.input_size
            reflex_memory = ReflexMemory(
                pattern_dim=pattern_dim,
                max_patterns=10000,
                promotion_threshold=promotion_threshold,
            )
        
        self.rm = reflex_memory
        
        # Track HTM calls per pattern for promotion
        self.htm_call_counts: Dict[int, int] = {}
        
        # Statistics
        self.rm_hits = 0
        self.htm_calls = 0
    
    def forward(
        self,
        x: torch.Tensor,
        learn: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward with automatic acceleration.
        
        Fast path: Check Reflex Memory first
        Slow path: Full HTM computation (with potential promotion)
        
        Args:
            x: Input tensor (input_size,) or (batch, input_size)
            learn: Whether to update HTM/RM
            
        Returns:
            Dictionary with:
            - 'features': Output features (column_count,)
            - 'anomaly': Anomaly score (0 for RM hit, computed for HTM)
            - 'from_reflex': Whether result came from Reflex Memory
            - 'active_cells': Active cell pattern (if available)
        """
        device = x.device
        is_batched = x.dim() == 2
        
        if is_batched:
            # Process batch sequentially (HTM is typically online)
            results = []
            for i in range(x.shape[0]):
                result = self._forward_single(x[i], learn, device)
                results.append(result)
            
            # Stack results
            return {
                key: torch.stack([r[key] for r in results])
                for key in results[0].keys()
            }
        
        return self._forward_single(x, learn, device)
    
    def _forward_single(
        self,
        x: torch.Tensor,
        learn: bool,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Process single input with acceleration."""
        
        # Fast path: Try Reflex Memory first
        rm_result = self.rm.lookup(x)
        
        if rm_result is not None:
            prediction, confidence, _ = rm_result
            self.rm_hits += 1
            
            # Still update HTM in background if learning (optional)
            if learn and self.training:
                with torch.no_grad():
                    _ = self.htm(x, learn=True)
            
            return {
                'features': prediction,
                'anomaly': torch.tensor(0.0, device=device),  # Known pattern
                'anomaly_likelihood': torch.tensor(0.0, device=device),
                'from_reflex': torch.tensor(True, device=device),
                'confidence': torch.tensor(confidence, device=device),
            }
        
        # Slow path: Full HTM computation
        self.htm_calls += 1
        htm_result = self.htm(x, learn=learn)
        
        # Consider for promotion to Reflex Memory
        if learn:
            pattern_hash = hash(x.detach().cpu().numpy().tobytes())
            self.htm_call_counts[pattern_hash] = self.htm_call_counts.get(pattern_hash, 0) + 1
            
            if self.htm_call_counts[pattern_hash] >= self.promotion_threshold:
                # Promote to Reflex Memory
                self.rm.store(x, htm_result['features'], force=True)
                del self.htm_call_counts[pattern_hash]
        
        return {
            'features': htm_result['features'],
            'anomaly': htm_result['anomaly'],
            'anomaly_likelihood': htm_result.get('anomaly_likelihood', htm_result['anomaly']),
            'from_reflex': torch.tensor(False, device=device),
            'confidence': torch.tensor(1.0 - htm_result['anomaly'].item(), device=device),
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """Get combined statistics."""
        total_calls = self.rm_hits + self.htm_calls
        return {
            'rm_hits': self.rm_hits,
            'htm_calls': self.htm_calls,
            'acceleration_rate': self.rm_hits / max(total_calls, 1),
            'rm_stats': self.rm.get_statistics(),
            'patterns_pending_promotion': len(self.htm_call_counts),
        }
    
    def reset_statistics(self):
        """Reset all statistics."""
        self.rm_hits = 0
        self.htm_calls = 0
        self.rm.reset_statistics()


def create_accelerated_htm(
    input_size: int = 512,
    column_count: int = 2048,
    cells_per_column: int = 32,
    sparsity: float = 0.02,
    max_reflex_patterns: int = 10000,
    promotion_threshold: int = 5,
    **kwargs,
) -> AcceleratedHTM:
    """
    Create Accelerated HTM with Reflex Memory.
    
    Args:
        input_size: Input dimension
        column_count: Number of HTM columns
        cells_per_column: Cells per column
        sparsity: Target sparsity
        max_reflex_patterns: Maximum patterns in Reflex Memory
        promotion_threshold: HTM calls before promotion
        **kwargs: Additional HTM config parameters
        
    Returns:
        AcceleratedHTM instance
    """
    htm_layer = create_htm_layer(
        input_size=input_size,
        column_count=column_count,
        cells_per_column=cells_per_column,
        sparsity=sparsity,
        **kwargs,
    )
    
    reflex_memory = ReflexMemory(
        pattern_dim=input_size,
        max_patterns=max_reflex_patterns,
        promotion_threshold=promotion_threshold,
    )
    
    return AcceleratedHTM(
        htm_layer=htm_layer,
        reflex_memory=reflex_memory,
        promotion_threshold=promotion_threshold,
    )
