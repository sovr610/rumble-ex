# HTM Temporal Memory -- Implementation Reference

Target audience: Claude Code agents implementing or upgrading the TM
algorithm in `brain_ai/temporal/htm.py`.

---

## 1. Overview

Temporal Memory (TM) learns temporal sequences of Sparse Distributed
Representations (SDRs). It does so by modelling **cells within columns** and
forming **dendritic segments** between cells in different columns.

Core ideas:

- Each column represents a feedforward input feature (from the Spatial
  Pooler). Multiple cells per column encode **context** -- the same column
  can be active, but which cell fires depends on preceding sequence history.
- Learning happens through **local Hebbian rules** applied to dendritic
  segments. There is no backpropagation, no optimizer, no global loss
  function.
- Segments connect a cell to a set of presynaptic cells. When enough
  connected presynaptic cells are active, the segment "matches" and puts its
  cell into a **predictive state** for the next timestep.
- Correctly predicted cells activate cleanly (no burst). Unpredicted columns
  **burst** (all cells fire), generating a high anomaly score.

Every timestep produces four outputs:

| Output | Shape | Description |
|---|---|---|
| `active_cells` | `(B, K_cells)` int | Cells that fired this step |
| `winner_cells` | `(B, K_active_cols)` int | One winning cell per active column |
| `predictive_cells` | `(B, K_pred)` int | Cells predicted for the **next** step |
| `anomaly_score` | `(B,)` float | Fraction of active columns that were not predicted |

---

## 2. Cell-Column Architecture

### 2.1 Dimensions

```
N_columns        -- number of minicolumns (e.g. 2048)
cells_per_column -- cells in each column  (e.g. 32)
N_cells          -- total cells = N_columns * cells_per_column
```

### 2.2 Addressing

Cells are numbered contiguously. Column `c` owns cells
`[c * cells_per_column, (c+1) * cells_per_column)`.

```python
def cell_id(column: int, offset: int, cpc: int) -> int:
    """Return global cell index."""
    return column * cpc + offset

def column_for_cell(cell: int, cpc: int) -> int:
    """Return column index that owns this cell."""
    return cell // cpc

def offset_in_column(cell: int, cpc: int) -> int:
    """Return intra-column offset (0..cpc-1)."""
    return cell % cpc
```

### 2.3 Vectorized Column Lookup

```python
# Given a tensor of cell indices  (any shape)
columns = cell_indices // cells_per_column   # column per cell
offsets = cell_indices %  cells_per_column    # offset per cell
```

### 2.4 Cells-for-Column Expansion

```python
def cells_for_columns(col_indices: Tensor, cpc: int) -> Tensor:
    """Expand column indices to all cell indices in those columns.

    col_indices: (K,) int
    returns:     (K * cpc,) int
    """
    base = col_indices.unsqueeze(-1) * cpc          # (K, 1)
    offsets = torch.arange(cpc, device=base.device)  # (cpc,)
    return (base + offsets).reshape(-1)
```

---

## 3. Core States (Per Step)

### 3.1 State Tensors

All states use **SDR index form** (sorted integer indices of active
elements) rather than dense boolean vectors. This saves memory when
sparsity is high (2--5 % typical).

| State | Typical shape | Content |
|---|---|---|
| `active_cells` | `(B, K_cells)` | Indices of all cells that fired |
| `winner_cells` | `(B, K_active_cols)` | One winner cell index per active column |
| `predictive_cells` | `(B, K_pred)` | Indices of cells with matching segments |
| `predicted_columns` | `(B, K_pred_cols)` | Unique columns containing predictive cells |

`K_*` varies per batch item. Pad with `-1` or use a list-of-tensors when
batch items have different counts.

### 3.2 Dense / Index Conversion

```python
def indices_to_dense(indices: Tensor, size: int) -> Tensor:
    """Convert index SDR to dense boolean mask.

    indices: (K,) int, values in [0, size)
    returns: (size,) bool
    """
    mask = torch.zeros(size, dtype=torch.bool, device=indices.device)
    if indices.numel() > 0:
        mask[indices] = True
    return mask

def dense_to_indices(mask: Tensor) -> Tensor:
    """Convert dense boolean mask to sorted index SDR.

    mask:    (size,) bool
    returns: (K,) int
    """
    return mask.nonzero(as_tuple=False).squeeze(-1)
```

### 3.3 Previous-Step Buffers

Learning requires access to the *previous* timestep's state. Maintain:

```python
prev_active_cells:  Tensor   # index form
prev_winner_cells:  Tensor   # index form
```

Copy current to previous **before** computing new activations each step.

---

## 4. Activation Logic (Per Active Column)

For each active column in the current SDR, determine which cells fire and
which cell wins.

### 4.1 Predicted Column (column has >= 1 predictive cell)

```
active_cells  <-- predicted cells in this column
winner_cell   <-- predicted cell with the strongest segment match
```

Only the predicted cells fire. No burst. The winner is the cell whose best
matching segment had the highest overlap with `prev_active_cells`.

### 4.2 Unpredicted Column -- BURST

```
active_cells  <-- ALL cells in this column
winner_cell   <-- cell with best matching segment to prev_active_cells
                  if no matching segment: cell with fewest existing segments
```

All `cells_per_column` cells fire (burst). The anomaly counter increments.

### 4.3 Anomaly Score

```python
burst_count       = count of active columns with zero predictive cells
total_active_cols = count of active columns
anomaly_score     = burst_count / max(total_active_cols, 1)
```

### 4.4 Vectorized Implementation

```python
def compute_activations(
    active_col_indices: Tensor,     # (K_cols,) int
    predictive_cells: Tensor,       # (K_pred,) int
    prev_active_cells: Tensor,      # (K_prev,) int
    seg_store: "SegmentStore",
    cells_per_column: int,
) -> Tuple[Tensor, Tensor, float]:
    """Compute active_cells, winner_cells, anomaly_score.

    Returns:
        active_cells:  (K_active,) int
        winner_cells:  (K_cols,) int   -- one per active column
        anomaly_score: float
    """
    cpc = cells_per_column

    # Build dense masks for fast set membership
    pred_mask = indices_to_dense(predictive_cells, N_cells)

    # Per-column: check if any cell is predicted
    # Expand columns to all cells, then check pred_mask
    all_col_cells = cells_for_columns(active_col_indices, cpc)  # (K_cols*cpc,)
    pred_in_cols = pred_mask[all_col_cells].reshape(-1, cpc)    # (K_cols, cpc)
    col_has_pred = pred_in_cols.any(dim=1)                       # (K_cols,) bool

    # --- Predicted columns ---
    # active_cells = predicted cells only
    # winner_cell  = cell with strongest match
    pred_col_mask = col_has_pred                                 # (K_cols,)

    # --- Bursting columns ---
    burst_col_mask = ~col_has_pred                               # (K_cols,)
    anomaly_score = burst_col_mask.float().mean().item()

    active_cells_list = []
    winner_cells = torch.empty(len(active_col_indices), dtype=torch.long,
                               device=active_col_indices.device)

    for i, col in enumerate(active_col_indices):
        start = col * cpc
        if col_has_pred[i]:
            # Predicted column: activate only predicted cells
            cell_range = torch.arange(start, start + cpc,
                                      device=col.device)
            pc = cell_range[pred_mask[start:start + cpc]]
            active_cells_list.append(pc)
            # Winner = predicted cell with strongest segment overlap
            winner_cells[i] = seg_store.best_cell_by_overlap(
                pc, prev_active_cells
            )
        else:
            # Bursting column: activate ALL cells
            cell_range = torch.arange(start, start + cpc,
                                      device=col.device)
            active_cells_list.append(cell_range)
            # Winner = cell with best matching segment, or fewest segments
            winner_cells[i] = seg_store.pick_burst_winner(
                cell_range, prev_active_cells
            )

    active_cells = torch.cat(active_cells_list)
    return active_cells, winner_cells, anomaly_score
```

---

## 5. Prediction Logic (Segment Matching)

A dendritic segment "matches" when enough of its connected presynaptic
cells are currently active.

### 5.1 Matching Condition

```python
connected_synapses  = synapse.perm >= connected_threshold
active_presynaptic  = synapse.src_cell IN active_cells_set
active_connected    = connected_synapses AND active_presynaptic
match_count         = sum(active_connected)
segment_matches     = match_count >= activation_threshold
```

### 5.2 Predictive Cell Rule

When a segment matches, its owning cell becomes **predictive** for the
**next** timestep. Multiple segments on the same cell can match; the cell
is predictive regardless of how many match.

### 5.3 Winner Selection Among Predicted Cells

When choosing the winner cell among multiple predicted cells in a column,
pick the cell whose strongest matching segment has the highest
`match_count`. Break ties arbitrarily (e.g., lowest cell index).

### 5.4 Vectorized Matching with CSR Store

See Section 6.2 for the full vectorized implementation using the flat
synapse arrays.

---

## 6. Sparse Segment Store (CSR-Style)

This is the critical engineering optimization. Replace the existing
`Dict[int, List[Dict[int, float]]]` with flat contiguous arrays amenable
to GPU scatter/gather operations.

### 6.1 Layout

```python
class SegmentStore:
    """CSR-style flat storage for dendritic segments and synapses."""

    # ----- Capacity constants -----
    S_max: int   # max number of segments (pre-allocated)
    M_max: int   # max number of synapses (pre-allocated)

    # ----- Segment metadata  (length S_max, live count = num_segments) -----
    seg_cell:   Tensor  # (S_max,) int32  -- owning cell for each segment
    seg_start:  Tensor  # (S_max,) int32  -- start index into synapse arrays
    seg_len:    Tensor  # (S_max,) int16  -- number of synapses in segment

    # ----- Synapse data  (length M_max, live count = num_synapses) -----
    syn_src_cell: Tensor  # (M_max,) int32   -- presynaptic cell id
    syn_perm:     Tensor  # (M_max,) float32 -- permanence value [0, 1]

    # ----- Per-cell index  (length N_cells) -----
    cell_seg_start: Tensor  # (N_cells,) int32 -- first segment index for cell
    cell_seg_count: Tensor  # (N_cells,) int16 -- number of segments for cell

    # ----- Live counters -----
    num_segments: int  # current number of live segments
    num_synapses: int  # current number of live synapses

    # ----- Free lists -----
    seg_free_list: List[int]  # freed segment slots available for reuse
    syn_free_list: List[int]  # freed synapse slots available for reuse
```

Register all tensors as **buffers** (`self.register_buffer`) so they
participate in `state_dict` serialization and `.to(device)` calls.

### 6.2 Vectorized Segment Matching

```python
def compute_matching_segments(
    self,
    active_cells: Tensor,        # (K,) int -- currently active cell indices
    connected_threshold: float,
    activation_threshold: int,
) -> Tuple[Tensor, Tensor]:
    """Find all segments that match (have enough active connected synapses).

    Returns:
        matching_seg_indices: (M_match,) int -- indices into seg_* arrays
        match_counts:         (M_match,) int -- overlap count per segment
    """
    S = self.num_segments
    M = self.num_synapses

    # Step 1: Build dense mask of active cells
    active_mask = torch.zeros(self.N_cells, dtype=torch.bool,
                              device=self.syn_src_cell.device)
    active_mask[active_cells] = True

    # Step 2: Per-synapse flags (only live synapses)
    connected = self.syn_perm[:M] >= connected_threshold    # (M,) bool
    active_pre = active_mask[self.syn_src_cell[:M]]         # (M,) bool
    active_connected = connected & active_pre               # (M,) bool

    # Step 3: Sum per segment using segment boundaries
    # Build segment-id for each synapse via searchsorted or repeat
    seg_ids = self._synapse_to_segment_id(M)                # (M,) int
    match_counts_all = torch.zeros(S, dtype=torch.int32,
                                   device=active_connected.device)
    match_counts_all.scatter_add_(
        0, seg_ids, active_connected.int()
    )

    # Step 4: Threshold
    matching_mask = match_counts_all >= activation_threshold  # (S,) bool
    matching_seg_indices = matching_mask.nonzero(as_tuple=False).squeeze(-1)
    match_counts = match_counts_all[matching_seg_indices]

    return matching_seg_indices, match_counts

def _synapse_to_segment_id(self, M: int) -> Tensor:
    """Map each synapse slot to its owning segment index.

    Uses seg_start and seg_len to build the mapping.
    Pre-compute and cache when segment layout changes.
    """
    S = self.num_segments
    seg_id = torch.zeros(M, dtype=torch.long,
                         device=self.seg_start.device)
    for s in range(S):
        start = self.seg_start[s].item()
        length = self.seg_len[s].item()
        seg_id[start:start + length] = s
    return seg_id
```

For production, replace the Python loop in `_synapse_to_segment_id` with
`torch.repeat_interleave`:

```python
def _synapse_to_segment_id_fast(self, S: int) -> Tensor:
    lengths = self.seg_len[:S].long()
    return torch.repeat_interleave(
        torch.arange(S, device=lengths.device), lengths
    )
```

### 6.3 Memory Management

#### Pre-allocation

Allocate arrays at construction time with generous upper bounds:

```python
S_max = N_cells * max_segments_per_cell     # e.g. 65536 * 128
M_max = S_max * max_synapses_per_segment    # e.g. 65536 * 128 * 32
```

For very large networks, start smaller and grow with `torch.cat` when
capacity is exceeded, doubling each time.

#### Free Lists

When a segment or synapse is deleted, append its index to the
corresponding free list. On allocation, pop from the free list first:

```python
def allocate_segment(self) -> int:
    if self.seg_free_list:
        return self.seg_free_list.pop()
    idx = self.num_segments
    self.num_segments += 1
    assert idx < self.S_max, "Segment capacity exceeded"
    return idx

def allocate_synapse_block(self, count: int) -> int:
    """Allocate a contiguous block of synapse slots.

    For simplicity, always append at the end. Free-list reuse
    is slot-by-slot for individual synapse updates.
    """
    start = self.num_synapses
    self.num_synapses += count
    assert self.num_synapses <= self.M_max, "Synapse capacity exceeded"
    return start
```

#### Compaction

When the ratio `len(seg_free_list) / num_segments > compaction_threshold`
(e.g., 0.3), compact arrays by removing gaps:

```python
def compact(self):
    """Remove gaps left by deleted segments/synapses.

    Rebuild contiguous arrays and reset free lists.
    """
    live_mask = self._compute_live_segment_mask()  # (S_max,) bool
    # Gather live segments
    live_indices = live_mask.nonzero(as_tuple=False).squeeze(-1)
    # Rebuild seg_cell, seg_start, seg_len from live segments
    # Rebuild syn_src_cell, syn_perm by concatenating live synapse blocks
    # Update cell_seg_start, cell_seg_count
    # Reset free lists to empty
    ...
```

#### Capacity Caps

- `max_segments_per_cell`: prevent runaway segment growth on heavily
  used cells (default 128). When exceeded, delete the segment with the
  lowest total permanence.
- `max_synapses_per_segment`: cap synapses per segment (default 32).
  When exceeded, delete the synapse with the lowest permanence.

### 6.4 Online Growth

Creating a new segment for a winner cell, connecting it to the previous
timestep's winner cells:

```python
def grow_segment(
    self,
    winner_cell: int,
    prev_winner_cells: Tensor,    # (K_prev,) int
    max_new_synapses: int,
    initial_permanence: float,
):
    """Create a new segment on winner_cell with synapses to prev winners."""
    # Check per-cell cap
    cell_count = self.cell_seg_count[winner_cell].item()
    if cell_count >= self.max_segments_per_cell:
        # Evict weakest segment on this cell
        self._evict_weakest_segment(winner_cell)

    # Allocate segment slot
    seg_idx = self.allocate_segment()
    self.seg_cell[seg_idx] = winner_cell

    # Sample presynaptic cells (up to max_new_synapses)
    K = min(max_new_synapses, prev_winner_cells.numel())
    if K == 0:
        self.seg_start[seg_idx] = self.num_synapses
        self.seg_len[seg_idx] = 0
        return seg_idx

    perm = torch.randperm(prev_winner_cells.numel(),
                          device=prev_winner_cells.device)[:K]
    src_cells = prev_winner_cells[perm]

    # Allocate synapse block
    syn_start = self.allocate_synapse_block(K)
    self.seg_start[seg_idx] = syn_start
    self.seg_len[seg_idx] = K

    # Fill synapse data
    self.syn_src_cell[syn_start:syn_start + K] = src_cells
    self.syn_perm[syn_start:syn_start + K] = initial_permanence

    # Update per-cell index
    self.cell_seg_count[winner_cell] += 1

    return seg_idx
```

### 6.5 Segment and Synapse Pruning

Apply periodically (e.g., every N steps) or inline during learning:

```python
def prune(self, prune_threshold: float = 0.01):
    """Remove dead synapses and empty segments."""
    S = self.num_segments

    for s in range(S):
        if self._is_deleted(s):
            continue
        start = self.seg_start[s].item()
        length = self.seg_len[s].item()
        if length == 0:
            self._delete_segment(s)
            continue

        # Find synapses below threshold
        block = self.syn_perm[start:start + length]
        dead = block < prune_threshold
        if dead.any():
            # Compact within segment: move live synapses to front
            live = ~dead
            live_count = live.sum().item()
            self.syn_src_cell[start:start + live_count] = \
                self.syn_src_cell[start:start + length][live]
            self.syn_perm[start:start + live_count] = \
                block[live]
            self.seg_len[s] = live_count

            # If segment is now empty, delete it
            if live_count == 0:
                self._delete_segment(s)

def _delete_segment(self, seg_idx: int):
    """Mark a segment as deleted and add to free list."""
    cell = self.seg_cell[seg_idx].item()
    self.cell_seg_count[cell] -= 1
    self.seg_cell[seg_idx] = -1   # sentinel
    self.seg_free_list.append(seg_idx)
```

For a fully vectorized approach, build a live mask and use masked scatter:

```python
def prune_vectorized(self, prune_threshold: float = 0.01):
    """Vectorized pruning -- zero out dead synapses, mark empty segments."""
    M = self.num_synapses
    dead = self.syn_perm[:M] < prune_threshold
    self.syn_perm[:M][dead] = 0.0
    self.syn_src_cell[:M][dead] = -1  # sentinel

    # Recount per-segment live synapses
    seg_ids = self._synapse_to_segment_id_fast(self.num_segments)
    live_per_seg = torch.zeros(self.num_segments, dtype=torch.int32,
                               device=dead.device)
    live_per_seg.scatter_add_(0, seg_ids, (~dead).int())
    # Mark empty segments for deletion
    empty = live_per_seg == 0
    # ... handle deletion and free-list updates
```

---

## 7. Online Learning Rules (Hebbian)

All learning is local and Hebbian. No gradient, no optimizer. Apply the
rules **after** computing activations for the current step, using
`prev_active_cells` and `prev_winner_cells` as the presynaptic context.

### 7.1 Correctly Predicted Column

The winner cell was predicted. Find its active (matching) segment and
reinforce:

```python
def reinforce_segment(
    self,
    seg_idx: int,
    prev_active_mask: Tensor,   # (N_cells,) bool -- previous active cells
    perm_inc: float,
    perm_dec: float,
):
    """Strengthen synapses to active presynaptic cells, weaken others."""
    start = self.seg_start[seg_idx].item()
    length = self.seg_len[seg_idx].item()

    src_cells = self.syn_src_cell[start:start + length]
    perms = self.syn_perm[start:start + length]

    # Which presynaptic cells were active?
    pre_active = prev_active_mask[src_cells]   # (length,) bool

    # Hebbian update
    perms[pre_active]  += perm_inc    # reinforce active connections
    perms[~pre_active] -= perm_dec    # weaken inactive connections
    perms.clamp_(0.0, 1.0)

    self.syn_perm[start:start + length] = perms
```

### 7.2 Bursting Column

The column was not predicted. The winner cell was selected by best
matching segment or fewest segments.

- If the winner cell has a matching segment (overlap >= `min_threshold`):
  reinforce that segment with `reinforce_segment()` and optionally grow
  new synapses to previously active cells not yet connected.
- If the winner cell has **no** matching segment: grow a brand-new segment
  connecting to `prev_winner_cells` (see Section 6.4).

```python
def learn_on_burst(
    self,
    winner_cell: int,
    prev_active_cells: Tensor,
    prev_winner_cells: Tensor,
    prev_active_mask: Tensor,
    perm_inc: float,
    perm_dec: float,
    max_new_synapses: int,
    initial_permanence: float,
):
    """Learning rule for a bursting column's winner cell."""
    best_seg, best_overlap = self.best_matching_segment(
        winner_cell, prev_active_mask, min_threshold=0
    )

    if best_seg is not None and best_overlap >= self.min_threshold:
        # Reinforce existing segment
        self.reinforce_segment(best_seg, prev_active_mask,
                               perm_inc, perm_dec)
        # Grow additional synapses to unconnected prev_winner_cells
        self.grow_synapses_on_segment(best_seg, prev_winner_cells,
                                      max_new_synapses, initial_permanence)
    else:
        # Create new segment
        self.grow_segment(winner_cell, prev_winner_cells,
                          max_new_synapses, initial_permanence)
```

### 7.3 Incorrectly Predicted Column (Punishment)

If a cell was predictive but its column did **not** become active, punish
the matching segments to reduce false predictions:

```python
def punish_predicted_inactive(
    self,
    predictive_cells: Tensor,       # (K_pred,) int
    active_col_indices: Tensor,     # (K_cols,) int
    prev_active_mask: Tensor,       # (N_cells,) bool
    predicted_dec: float,
):
    """Punish segments that caused incorrect predictions."""
    # Find predicted columns
    pred_columns = predictive_cells // self.cells_per_column
    pred_columns_unique = pred_columns.unique()

    # Active columns as set
    active_col_set = set(active_col_indices.tolist())

    # Incorrectly predicted columns
    incorrect_cols = [c.item() for c in pred_columns_unique
                      if c.item() not in active_col_set]

    for col in incorrect_cols:
        # Find all predictive cells in this column
        start_cell = col * self.cells_per_column
        end_cell = start_cell + self.cells_per_column
        col_pred_cells = predictive_cells[
            (predictive_cells >= start_cell) & (predictive_cells < end_cell)
        ]

        for cell in col_pred_cells:
            # Find matching segments and punish
            seg_indices = self.get_segments_for_cell(cell.item())
            for seg_idx in seg_indices:
                start = self.seg_start[seg_idx].item()
                length = self.seg_len[seg_idx].item()
                src = self.syn_src_cell[start:start + length]
                pre_active = prev_active_mask[src]

                # Only punish synapses to cells that were active
                # (those are the ones that caused the false prediction)
                active_syn = start + pre_active.nonzero(
                    as_tuple=False
                ).squeeze(-1)
                self.syn_perm[active_syn] -= predicted_dec
                self.syn_perm[active_syn].clamp_(min=0.0)
```

### 7.4 Vectorized Batch Learning

Combine the three cases into one pass per batch item:

```python
def learn_step(
    self,
    active_col_indices: Tensor,     # (K_cols,) int
    winner_cells: Tensor,           # (K_cols,) int
    predictive_cells: Tensor,       # (K_pred,) int
    prev_active_cells: Tensor,      # (K_prev_a,) int
    prev_winner_cells: Tensor,      # (K_prev_w,) int
    prev_active_mask: Tensor,       # (N_cells,) bool
    perm_inc: float,
    perm_dec: float,
    predicted_dec: float,
    max_new_synapses: int,
    initial_permanence: float,
):
    """Full learning pass for one timestep."""
    pred_mask = indices_to_dense(predictive_cells, self.N_cells)

    for i, col in enumerate(active_col_indices):
        cell = winner_cells[i].item()
        col_start = col.item() * self.cells_per_column
        col_end = col_start + self.cells_per_column
        col_pred = pred_mask[col_start:col_end]

        if col_pred.any():
            # Case 7.1: correctly predicted -- reinforce
            best_seg, _ = self.best_matching_segment(
                cell, prev_active_mask, min_threshold=0
            )
            if best_seg is not None:
                self.reinforce_segment(best_seg, prev_active_mask,
                                       perm_inc, perm_dec)
        else:
            # Case 7.2: burst -- reinforce or grow
            self.learn_on_burst(
                cell, prev_active_cells, prev_winner_cells,
                prev_active_mask, perm_inc, perm_dec,
                max_new_synapses, initial_permanence
            )

    # Case 7.3: punish incorrect predictions
    self.punish_predicted_inactive(
        predictive_cells, active_col_indices,
        prev_active_mask, predicted_dec
    )
```

---

## 8. SequenceOutput Contract

Every call to `temporal_memory.step()` must return a `SequenceOutput`:

```python
@dataclass
class SequenceOutput:
    sdr: Tensor
    """(B, K) int -- active column indices this step."""

    pred_sdr: Tensor
    """(B, K_pred) int -- predicted column indices for next step."""

    anomaly_score: Tensor
    """(B,) float in [0, 1] -- fraction of unpredicted active columns."""

    aux: dict
    """Debug/diagnostic info, not used in forward path."""
```

### 8.1 Required `aux` Fields

| Key | Type | Description |
|---|---|---|
| `burst_count` | `Tensor (B,)` int | Number of bursting columns |
| `overlap_with_prediction` | `Tensor (B,)` int | Active columns that were predicted |
| `active_cell_count` | `Tensor (B,)` int | Total active cells |
| `segment_stats` | `dict` | `num_segments`, `num_synapses`, `avg_synapses_per_seg` |
| `duty_cycles` | `Tensor (N_cols,)` float | Per-column activation frequency (EMA) |

### 8.2 Computing `pred_sdr` from `predictive_cells`

```python
pred_columns = predictive_cells // cells_per_column
pred_sdr = pred_columns.unique(sorted=True)
```

If working in batched mode with padding:

```python
# For each batch item
pred_sdrs = []
for b in range(B):
    pc = predictive_cells[b]
    pc = pc[pc >= 0]  # remove padding
    cols = (pc // cells_per_column).unique()
    pred_sdrs.append(cols)
```

---

## 9. Anomaly Scoring

### 9.1 Raw Anomaly

```python
predicted_cols = (predictive_cells // cells_per_column).unique()
active_cols    = active_col_indices
overlap        = set_intersection(predicted_cols, active_cols)
anomaly        = 1.0 - len(overlap) / max(len(active_cols), 1)
```

Vectorized set intersection:

```python
def sdr_overlap_count(a: Tensor, b: Tensor) -> int:
    """Count elements present in both sorted index tensors."""
    if a.numel() == 0 or b.numel() == 0:
        return 0
    combined = torch.cat([a, b])
    uniques, counts = combined.unique(return_counts=True)
    return (counts > 1).sum().item()
```

### 9.2 Anomaly Likelihood (Rolling Gaussian)

Smooth the raw anomaly with a rolling window to distinguish "expected
anomaly" from "surprising anomaly":

```python
class AnomalyLikelihood:
    """Rolling Gaussian anomaly likelihood estimator."""

    def __init__(self, window_size: int = 1000, eps: float = 1e-6):
        self.window_size = window_size
        self.eps = eps
        self.history: List[float] = []

    def compute(self, raw_anomaly: float) -> float:
        """Return anomaly likelihood in [0, 1].

        High value = anomaly is unusual given recent history.
        Low value  = anomaly is typical (not surprising).
        """
        self.history.append(raw_anomaly)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        if len(self.history) < 10:
            return raw_anomaly  # not enough data

        mean = sum(self.history) / len(self.history)
        var = sum((x - mean) ** 2 for x in self.history) / len(self.history)
        std = max(var ** 0.5, self.eps)

        # One-sided: how far above the mean?
        z = (raw_anomaly - mean) / std
        # Approximate survival function:  1 - Phi(z)
        # Using logistic approximation for speed
        likelihood = 1.0 / (1.0 + math.exp(-1.7 * z))
        return float(min(max(likelihood, 0.0), 1.0))
```

### 9.3 Vectorized Batch Anomaly

```python
def batch_anomaly(
    active_cols_list: List[Tensor],
    pred_cols_list: List[Tensor],
) -> Tensor:
    """Compute anomaly score for each item in a batch.

    Returns: (B,) float tensor.
    """
    scores = []
    for active, pred in zip(active_cols_list, pred_cols_list):
        overlap = sdr_overlap_count(active, pred)
        n_active = max(active.numel(), 1)
        scores.append(1.0 - overlap / n_active)
    return torch.tensor(scores, dtype=torch.float32)
```

---

## 10. Mixed-Precision Safety

### 10.1 Type Requirements

| Data | Required dtype | Rationale |
|---|---|---|
| Cell indices | `int32` | Must be exact; `int16` overflows at 65536 cells |
| Synapse permanences (storage) | `float32` or `float16` | Can store in fp16 to halve memory |
| Synapse permanences (update) | `float32` | Increments of 0.01--0.1 lose precision in fp16 |
| Segment match counts | `int32` | Integer counts, no precision issue |
| Anomaly scores | `float32` | Continuous values, fp16 is acceptable but not necessary |
| `scatter_add_` accumulator | `float32` | fp16 scatter_add can lose precision with many terms |

### 10.2 Safe Update Pattern

```python
# If permanences are stored in fp16 for memory savings:
def update_permanences_mixed(self, seg_idx, prev_active_mask, inc, dec):
    start = self.seg_start[seg_idx].item()
    length = self.seg_len[seg_idx].item()

    # Upcast to fp32 for the update
    perms = self.syn_perm[start:start + length].float()
    src = self.syn_src_cell[start:start + length]
    active = prev_active_mask[src]

    perms[active]  += inc
    perms[~active] -= dec
    perms.clamp_(0.0, 1.0)

    # Downcast back to fp16 for storage
    self.syn_perm[start:start + length] = perms.half()
```

### 10.3 AMP Context

When used inside `torch.cuda.amp.autocast`, segment matching and learning
should be wrapped in `with torch.cuda.amp.autocast(enabled=False):` to
prevent unintended fp16 downcasting of permanence updates.

---

## 11. Migration from Existing Code

### 11.1 Current Implementation

The existing `PytorchTemporalMemory` class in `brain_ai/temporal/htm.py`
(lines 243--530) has the following structure:

```
segments: Dict[int, List[Dict[int, float]]]
```

Each entry maps `cell_id -> list of segments`, where each segment is a
`Dict[presynaptic_cell_id -> permanence_float]`.

Problems:

1. **Not vectorizable.** Every segment match requires Python-level loops
   over dicts. Cannot move to GPU.
2. **O(N) Python iteration** for matching. With 65536 cells and 128
   segments each, this is millions of Python dict lookups per step.
3. **Not serializable as state_dict.** The `segments` dict is not a
   buffer or parameter, so `model.state_dict()` does not capture learned
   segments. Checkpointing is broken.
4. **No batch support.** The forward pass processes one item at a time.
   State is shared across batch items in an undefined way.
5. **Dense state buffers.** `active_cells`, `winner_cells`,
   `predictive_cells` are stored as `(N_cells,)` dense float tensors.
   With 65536 cells and 2% sparsity, this wastes 98% of memory.

### 11.2 Required Changes

| Aspect | Old | New |
|---|---|---|
| Segment storage | `Dict[int, List[Dict]]` | CSR flat arrays (Section 6) |
| Segment matching | Python for-loop over dicts | `scatter_add_` / `segment_reduce` |
| Memory management | Unbounded dict growth | Pre-allocated + free-list |
| State serialization | Not in state_dict | All arrays as `register_buffer` |
| Cell states | Dense `(N_cells,)` float | Index SDR `(K,)` int |
| Return type | `Dict[str, Tensor]` | `SequenceOutput` dataclass |
| Batch support | Sequential loop | Proper per-item state with padding |
| Pruning | Manual `del` in dict | Vectorized threshold + free-list |

### 11.3 Migration Steps

1. Define `SegmentStore` class with CSR layout (Section 6.1).
2. Implement `compute_matching_segments` (Section 6.2).
3. Implement `grow_segment` and `reinforce_segment` (Sections 6.4, 7.1).
4. Implement `prune` (Section 6.5).
5. Rewrite `compute_activity` to use `SegmentStore` matching instead of
   dict iteration (Section 4.4).
6. Rewrite `learn` to call `learn_step` (Section 7.4).
7. Change return type to `SequenceOutput` (Section 8).
8. Convert dense cell-state buffers to index-form SDRs (Section 3).
9. Add `AnomalyLikelihood` estimator (Section 9.2).
10. Register all store arrays as buffers.
11. Update `HTMLayer._forward_pytorch` to handle `SequenceOutput`.
12. Add unit tests comparing old and new implementations on short
    sequences (verify identical anomaly scores for identical inputs).

### 11.4 Backward Compatibility

Provide a shim so existing code that expects the old dict return format
still works:

```python
def forward(self, active_columns, learn=True):
    seq_out = self.step(active_columns, learn=learn)
    # Legacy dict format
    return {
        'active_cells': indices_to_dense(seq_out.sdr, self.N_cells),
        'predictive_cells': indices_to_dense(seq_out.pred_sdr, self.N_cells),
        'anomaly': seq_out.anomaly_score,
    }
```

---

## 12. Anti-Patterns

Avoid the following mistakes when implementing or modifying TM.

### 12.1 Dict-Based Segments

```python
# BAD: O(N) Python loops, not GPU-friendly
segments: Dict[int, List[Dict[int, float]]] = {}
```

Use CSR flat arrays (Section 6) instead. Dict-based storage is acceptable
only for prototyping with < 1000 cells.

### 12.2 Dense Cell-to-Cell Connectivity

```python
# BAD: N_cells x N_cells matrix. With 65536 cells = 16 GB float32.
connectivity = torch.zeros(N_cells, N_cells)
```

TM connectivity is extremely sparse. Use CSR or COO sparse formats.

### 12.3 Gradient-Based Learning in TM

```python
# BAD: TM uses pure Hebbian rules, not backprop
optimizer = torch.optim.Adam(tm.parameters())
loss.backward()
optimizer.step()
```

TM permanences are **not** `nn.Parameter`. They are buffers updated by
Hebbian rules. Do not register them as parameters. Do not pass them to an
optimizer.

### 12.4 Forgetting `winner_cells` Between Steps

```python
# BAD: resets learning context
def step(self, ...):
    self.prev_winner_cells = None   # WRONG
    ...
```

`prev_winner_cells` must persist between calls to `step()`. New segments
connect to the previous step's winner cells. If cleared, all new
segments will have zero synapses and learning will fail silently.

### 12.5 Resetting State Mid-Sequence

```python
# BAD: destroys temporal context
for t in range(seq_len):
    tm.reset()          # WRONG -- do not reset inside a sequence
    tm.step(sdr[t])
```

Call `reset()` only **between** sequences (e.g., between episodes or
documents). Within a sequence, state must accumulate.

### 12.6 Per-Batch-Item Segment Store

```python
# BAD: segments should be shared across time, not per-batch-item
for b in range(batch_size):
    stores[b] = SegmentStore()   # WRONG -- one store per batch item
```

The segment store is the **learned model**. It accumulates knowledge over
many sequences. There is one store shared across all forward calls. The
per-step **state** (active_cells, winner_cells, predictive_cells) varies
per batch item, but the segments do not.

### 12.7 Ignoring Segment Capacity Limits

```python
# BAD: unbounded segment growth will exhaust memory
def grow_segment(self, cell, ...):
    # No check on max_segments_per_cell
    seg_idx = self.allocate_segment()
    ...
```

Always enforce `max_segments_per_cell` and `max_synapses_per_segment`.
When limits are hit, evict the weakest (lowest total permanence) before
creating new ones.

### 12.8 Using `torch.unique` Without Sorting

```python
# CAUTION: torch.unique with sorted=False gives non-deterministic order
cols = predictive_cells // cpc
unique_cols = cols.unique(sorted=False)  # order varies across runs
```

Use `sorted=True` (the default) to ensure deterministic behavior across
runs and devices.

---

## Quick Reference: Default Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `cells_per_column` | 32 | Context depth per column |
| `activation_threshold` | 13 | Synapses needed for segment match |
| `min_threshold` | 10 | Synapses needed for "best matching" (learning) |
| `max_new_synapses` | 20 | Synapses per new segment |
| `initial_permanence` | 0.21 | Starting permanence for new synapses |
| `connected_threshold` | 0.50 | Permanence >= this means "connected" |
| `perm_inc` | 0.10 | Hebbian reinforcement increment |
| `perm_dec` | 0.10 | Hebbian weakening decrement |
| `predicted_dec` | 0.05 | Punishment decrement for false predictions (more aggressive than Numenta's 0.004 for faster adaptation) |
| `max_segments_per_cell` | 128 | Cap on segments per cell |
| `max_synapses_per_segment` | 32 | Cap on synapses per segment |
| `prune_threshold` | 0.01 | Permanences below this are pruned |
| `anomaly_window` | 1000 | Rolling window for anomaly likelihood |

---

## Quick Reference: Step-by-Step Execution Order

```
1. Copy current state to previous:
     prev_active_cells  = active_cells
     prev_winner_cells  = winner_cells

2. Receive active_columns (SDR from Spatial Pooler)

3. Compute activations (Section 4):
     For each active column:
       if column has predictive cells:  activate predicted cells only
       else (burst):                    activate all cells in column
     Record winner_cells (one per active column)

4. Compute anomaly (Section 9):
     anomaly = 1 - overlap(predicted_cols, active_cols) / |active_cols|

5. Learn (Section 7):
     For each winner cell:
       if column was predicted:  reinforce matching segment
       if column burst:          reinforce or grow segment
     Punish incorrectly predicted cells

6. Compute predictions for NEXT step (Section 5):
     For each segment in store:
       if segment matches against current active_cells:
         mark owning cell as predictive

7. Prune (Section 6.5):
     Remove synapses with perm < prune_threshold
     Remove empty segments

8. Return SequenceOutput (Section 8)
```
