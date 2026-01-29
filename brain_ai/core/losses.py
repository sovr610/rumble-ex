"""
SNN Training Losses and Regularization

Implements specialized loss functions for Spiking Neural Networks:
- ProbSpikes loss: Cross-entropy on normalized spike counts
- Spike rate regularization: Enforce biologically plausible firing rates
- Temporal consistency loss: Encourage stable temporal patterns
- Inter-spike interval loss: Leverage precise spike timing

Based on latest research (2025):
- Yu et al. "Beyond Rate Coding: Surrogate Gradients Enable Spike Timing Learning"
- Shrestha et al. "ProbSpikes loss for SNN training"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


def prob_spikes_loss(
    spike_output: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    ProbSpikes loss - cross-entropy on normalized spike counts.
    
    Instead of using final membrane potential, converts spike counts
    to a probability distribution and computes cross-entropy.
    This encourages the correct class to emit the largest fraction of spikes.
    
    Args:
        spike_output: (time, batch, num_classes) spike tensor
        targets: (batch,) class labels
        temperature: Softmax temperature for probability conversion
        eps: Small epsilon for numerical stability
        
    Returns:
        Loss value (scalar)
        
    Reference:
        Shrestha et al. (2022) - SNNs with learnable delays
    """
    # Sum spikes over time
    spike_counts = spike_output.sum(dim=0)  # (batch, num_classes)
    
    # Convert to log probabilities with temperature
    log_probs = F.log_softmax(spike_counts / temperature, dim=-1)
    
    # Cross-entropy loss
    loss = F.nll_loss(log_probs, targets)
    
    return loss


def spike_rate_regularization(
    spikes: torch.Tensor,
    target_rate: float = 0.1,
    rate_type: str = "l2",
) -> torch.Tensor:
    """
    Regularize spike rates to biologically plausible levels.
    
    Cortical neurons typically fire at 10-20 Hz (roughly 10% of timesteps
    in typical SNN simulations). This regularization encourages sparse,
    biologically realistic firing patterns.
    
    Args:
        spikes: (time, batch, neurons) or (batch, time, neurons) spike tensor
        target_rate: Target firing rate (0.1 = 10% of timesteps)
        rate_type: "l2" for squared penalty, "l1" for absolute penalty
        
    Returns:
        Regularization loss (scalar)
    """
    # Compute firing rate per neuron (average over time)
    if spikes.dim() == 3:
        rates = spikes.mean(dim=0)  # Average over time -> (batch, neurons)
    else:
        rates = spikes.mean()  # Overall rate
    
    # Penalty for deviation from target
    if rate_type == "l2":
        rate_loss = ((rates - target_rate) ** 2).mean()
    elif rate_type == "l1":
        rate_loss = (rates - target_rate).abs().mean()
    else:
        raise ValueError(f"Unknown rate_type: {rate_type}")
    
    return rate_loss


def spike_rate_range_regularization(
    spikes: torch.Tensor,
    min_rate: float = 0.01,
    max_rate: float = 0.3,
) -> torch.Tensor:
    """
    Regularize spike rates to be within a reasonable range.
    
    Penalizes neurons that are:
    - Too silent (rate < min_rate): May be dead neurons
    - Too active (rate > max_rate): Unbiological and energy-inefficient
    
    Args:
        spikes: Spike tensor (time, batch, neurons)
        min_rate: Minimum acceptable firing rate
        max_rate: Maximum acceptable firing rate
        
    Returns:
        Regularization loss
    """
    rates = spikes.mean(dim=0)  # (batch, neurons)
    
    # Penalty for being below minimum
    below_min = F.relu(min_rate - rates)
    
    # Penalty for being above maximum  
    above_max = F.relu(rates - max_rate)
    
    return (below_min + above_max).mean()


def temporal_consistency_loss(
    spikes: torch.Tensor,
    window_size: int = 5,
) -> torch.Tensor:
    """
    Encourage consistent temporal patterns within time windows.
    
    Computes variance of spike patterns within sliding windows,
    encouraging the network to develop stable firing patterns.
    
    Args:
        spikes: (time, batch, neurons) spike tensor
        window_size: Size of sliding window
        
    Returns:
        Consistency loss (lower is more consistent)
    """
    time_steps = spikes.shape[0]
    if time_steps < window_size * 2:
        return torch.tensor(0.0, device=spikes.device)
    
    # Compute windowed spike rates
    windows = []
    for i in range(time_steps - window_size + 1):
        window = spikes[i:i + window_size].mean(dim=0)  # (batch, neurons)
        windows.append(window)
    
    windows = torch.stack(windows)  # (num_windows, batch, neurons)
    
    # Variance across windows (lower = more consistent)
    variance = windows.var(dim=0).mean()
    
    return variance


def temporal_sparsity_loss(
    spikes: torch.Tensor,
    target_temporal_sparsity: float = 0.8,
) -> torch.Tensor:
    """
    Encourage temporal sparsity - neurons should not fire every timestep.
    
    Temporal sparsity = fraction of timesteps with NO spike.
    High temporal sparsity (0.8-0.9) is biologically realistic.
    
    Args:
        spikes: (time, batch, neurons) spike tensor
        target_temporal_sparsity: Target fraction of silent timesteps
        
    Returns:
        Sparsity loss
    """
    # For each neuron, compute fraction of silent timesteps
    silent_timesteps = (spikes.sum(dim=-1) == 0).float()  # (time, batch)
    temporal_sparsity = silent_timesteps.mean()
    
    # L2 penalty for deviation from target
    loss = (temporal_sparsity - target_temporal_sparsity) ** 2
    
    return loss


def inter_spike_interval_loss(
    spikes: torch.Tensor,
    target_cv: float = 1.0,
) -> torch.Tensor:
    """
    Regularize inter-spike intervals toward target coefficient of variation.
    
    Coefficient of Variation (CV) = std(ISIs) / mean(ISIs)
    - CV ≈ 1: Poisson-like firing (biologically realistic)
    - CV < 1: Too regular (non-biological)
    - CV > 1: Too bursty
    
    Args:
        spikes: (time, batch, neurons) spike tensor
        target_cv: Target coefficient of variation (1.0 for Poisson-like)
        
    Returns:
        ISI regularization loss
    """
    time_steps, batch_size, num_neurons = spikes.shape
    device = spikes.device
    
    total_loss = torch.tensor(0.0, device=device)
    count = 0
    
    # Process per neuron (vectorized would be better but complex)
    for b in range(min(batch_size, 4)):  # Sample for efficiency
        for n in range(min(num_neurons, 100)):  # Sample neurons
            spike_times = torch.where(spikes[:, b, n] > 0)[0].float()
            
            if len(spike_times) >= 3:
                # Compute ISIs
                isis = spike_times[1:] - spike_times[:-1]
                
                # Coefficient of variation
                mean_isi = isis.mean()
                std_isi = isis.std()
                
                if mean_isi > 0:
                    cv = std_isi / mean_isi
                    total_loss += (cv - target_cv) ** 2
                    count += 1
    
    if count > 0:
        return total_loss / count
    return torch.tensor(0.0, device=device)


def membrane_potential_regularization(
    membrane: torch.Tensor,
    max_membrane: float = 5.0,
) -> torch.Tensor:
    """
    Prevent membrane potential explosion.
    
    High membrane potentials can lead to gradient instability.
    This regularization keeps potentials bounded.
    
    Args:
        membrane: Membrane potential tensor
        max_membrane: Maximum allowed membrane potential
        
    Returns:
        Regularization loss
    """
    # Penalize potentials above max
    excess = F.relu(membrane.abs() - max_membrane)
    return excess.mean()


class SNNLoss(nn.Module):
    """
    Combined SNN loss with all regularization terms.
    
    Usage:
        loss_fn = SNNLoss(
            task_loss_weight=1.0,
            spike_rate_weight=0.1,
            temporal_weight=0.01,
        )
        
        loss, metrics = loss_fn(spikes, membrane, targets)
    """
    
    def __init__(
        self,
        task_loss_weight: float = 1.0,
        spike_rate_weight: float = 0.1,
        target_spike_rate: float = 0.1,
        temporal_weight: float = 0.01,
        temporal_sparsity_weight: float = 0.01,
        membrane_reg_weight: float = 0.001,
        use_prob_spikes: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        
        self.task_loss_weight = task_loss_weight
        self.spike_rate_weight = spike_rate_weight
        self.target_spike_rate = target_spike_rate
        self.temporal_weight = temporal_weight
        self.temporal_sparsity_weight = temporal_sparsity_weight
        self.membrane_reg_weight = membrane_reg_weight
        self.use_prob_spikes = use_prob_spikes
        self.temperature = temperature
    
    def forward(
        self,
        spikes: torch.Tensor,
        membrane: torch.Tensor,
        targets: torch.Tensor,
        return_metrics: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        Compute combined SNN loss.
        
        Args:
            spikes: (time, batch, num_classes) output spikes
            membrane: Final membrane potential
            targets: Class labels
            return_metrics: If True, return dict of individual losses
            
        Returns:
            total_loss: Combined loss scalar
            metrics: Dict of individual loss components (if return_metrics)
        """
        metrics = {}
        
        # 1. Task loss (classification)
        if self.use_prob_spikes:
            task_loss = prob_spikes_loss(spikes, targets, self.temperature)
        else:
            # Traditional: use final spike count
            spike_counts = spikes.sum(dim=0)
            task_loss = F.cross_entropy(spike_counts, targets)
        
        metrics['task_loss'] = task_loss.item()
        total_loss = self.task_loss_weight * task_loss
        
        # 2. Spike rate regularization
        if self.spike_rate_weight > 0:
            rate_loss = spike_rate_regularization(
                spikes, target_rate=self.target_spike_rate
            )
            metrics['spike_rate_loss'] = rate_loss.item()
            total_loss += self.spike_rate_weight * rate_loss
        
        # 3. Temporal consistency
        if self.temporal_weight > 0:
            temp_loss = temporal_consistency_loss(spikes)
            metrics['temporal_loss'] = temp_loss.item()
            total_loss += self.temporal_weight * temp_loss
        
        # 4. Temporal sparsity
        if self.temporal_sparsity_weight > 0:
            sparsity_loss = temporal_sparsity_loss(spikes)
            metrics['sparsity_loss'] = sparsity_loss.item()
            total_loss += self.temporal_sparsity_weight * sparsity_loss
        
        # 5. Membrane regularization
        if self.membrane_reg_weight > 0 and membrane is not None:
            mem_loss = membrane_potential_regularization(membrane)
            metrics['membrane_loss'] = mem_loss.item()
            total_loss += self.membrane_reg_weight * mem_loss
        
        # Compute additional metrics (not used in loss)
        with torch.no_grad():
            actual_rate = spikes.mean().item()
            metrics['actual_spike_rate'] = actual_rate
            
            # Accuracy
            pred = spikes.sum(dim=0).argmax(dim=-1)
            acc = (pred == targets).float().mean().item()
            metrics['accuracy'] = acc
        
        if return_metrics:
            return total_loss, metrics
        return total_loss, None


def compute_snn_metrics(
    spikes: torch.Tensor,
    membrane: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute comprehensive SNN metrics for monitoring.
    
    Args:
        spikes: (time, batch, output) spike tensor
        membrane: Final membrane potential
        targets: Class labels
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    with torch.no_grad():
        # 1. Basic accuracy
        spike_counts = spikes.sum(dim=0)
        predictions = spike_counts.argmax(dim=-1)
        metrics['accuracy'] = (predictions == targets).float().mean().item()
        
        # 2. Spike statistics
        metrics['spike_rate'] = spikes.mean().item()
        metrics['spike_rate_std'] = spikes.mean(dim=(0, 1)).std().item()
        
        # 3. Temporal statistics
        # Sparsity: fraction of (time, batch) pairs with no spikes
        silent = (spikes.sum(dim=-1) == 0).float()
        metrics['temporal_sparsity'] = silent.mean().item()
        
        # 4. Per-class spike analysis
        for c in range(min(spikes.shape[-1], 10)):  # First 10 classes
            class_mask = targets == c
            if class_mask.any():
                class_spikes = spike_counts[class_mask, c]
                metrics[f'class_{c}_spikes'] = class_spikes.mean().item()
        
        # 5. Entropy of output distribution
        probs = F.softmax(spike_counts, dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
        metrics['output_entropy'] = entropy.item()
        
        # 6. Confidence (max probability)
        confidence = probs.max(dim=-1).values.mean()
        metrics['confidence'] = confidence.item()
        
        # 7. Membrane statistics
        if membrane is not None:
            metrics['membrane_mean'] = membrane.mean().item()
            metrics['membrane_std'] = membrane.std().item()
            metrics['membrane_max'] = membrane.abs().max().item()
        
        # 8. Dead neurons (never fire)
        total_spikes_per_neuron = spikes.sum(dim=(0, 1))
        dead_neurons = (total_spikes_per_neuron == 0).float().mean()
        metrics['dead_neuron_fraction'] = dead_neurons.item()
        
        # 9. Saturated neurons (fire every timestep)
        max_possible = spikes.shape[0] * spikes.shape[1]
        saturated = (total_spikes_per_neuron >= max_possible * 0.9).float().mean()
        metrics['saturated_neuron_fraction'] = saturated.item()
    
    return metrics
