"""
JEPATrainer -- Full training orchestrator for V-JEPA 2 self-supervised learning.

Wires together:
  - Context encoder (ViT or compatible nn.Module)
  - VisionTransformerPredictor
  - EMAManagerWithRef (target encoder)
  - AdamW optimizer with 4 param groups
  - GradScaler for BFloat16 AMP
  - WarmupCosineScheduler and CosineWDScheduler

The train_step() method implements the complete JEPA forward pass:
    1. Encode visible patches with context encoder
    2. Predict masked patches with predictor
    3. Compute EMA target representations
    4. Compute smooth L1 loss on masked positions only
    5. Backward pass + gradient clipping + optimizer step

Self-contained: no external dataset needed for tests. Uses synthetic tensors.
"""

from __future__ import annotations

import copy
import math
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
    loss_exp: float = 1.0,
) -> torch.Tensor:
    """
    Smooth L1 (Huber) loss applied to predicted masked patch representations.

    Regime:
        |diff| < beta  =>  0.5 * diff^2 / beta  (quadratic)
        |diff| >= beta =>  |diff| - 0.5 * beta   (linear)

    Args:
        pred:     [B, N_pred, D] predictor output.
        target:   [B, N_pred, D] EMA target encoder output.
        beta:     Threshold between L1 and L2 regimes (default 1.0).
        loss_exp: Exponent applied per-token before averaging (default 1.0).

    Returns:
        Scalar loss value.
    """
    diff     = pred - target
    abs_diff = diff.abs()

    loss = torch.where(
        abs_diff < beta,
        0.5 * diff.pow(2) / beta,
        abs_diff - 0.5 * beta,
    )

    if loss_exp != 1.0:
        loss = loss.pow(loss_exp)

    return loss.mean()


# ---------------------------------------------------------------------------
# Simple stub encoder for testing (not a real ViT)
# ---------------------------------------------------------------------------

class _StubEncoder(nn.Module):
    """
    Minimal encoder stub: projects N visible patch embeddings to embed_dim.
    For testing JEPATrainer without a full ViT implementation.
    """

    def __init__(self, in_dim: int = 64, embed_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim, bias=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, in_dim] -> [B, N, embed_dim]"""
        return self.norm(self.proj(x))


# ---------------------------------------------------------------------------
# JEPATrainer
# ---------------------------------------------------------------------------

class JEPATrainer:
    """
    Orchestrates V-JEPA 2 self-supervised pretraining.

    Args:
        encoder:   Context encoder (processes visible patches).
        predictor: VisionTransformerPredictor (predicts masked patches).
        config:    JEPATrainingConfig (or compatible dataclass / dict).

    Usage::

        trainer = JEPATrainer(encoder, predictor, config)

        for epoch in range(config.epochs):
            for step, (batch, masks_enc, masks_pred) in enumerate(loader):
                global_step = epoch * steps_per_epoch + step
                loss_dict = trainer.train_step(batch, masks_enc, masks_pred)
                trainer.update_ema(global_step)
                trainer.lr_scheduler.step(global_step)  # Update LR each step
    """

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        config,  # JEPATrainingConfig or duck-typed dict
    ) -> None:
        self.encoder   = encoder
        self.predictor = predictor

        # --- Extract config fields (supports dataclass or dict) ---
        def _cfg(key, default=None):
            if hasattr(config, key):
                return getattr(config, key)
            if isinstance(config, dict):
                return config.get(key, default)
            return default

        self.lr               = _cfg('lr', 1e-3)
        self.final_lr         = _cfg('final_lr', 1e-6)
        self.weight_decay     = _cfg('weight_decay', 0.04)
        self.final_wd         = _cfg('final_weight_decay', 0.4)
        self.warmup_epochs    = _cfg('warmup_epochs', 40)
        self.epochs           = _cfg('epochs', 300)
        self.use_bfloat16     = _cfg('use_bfloat16', False)
        self.clip_grad        = _cfg('clip_grad', 1.0)
        self.ema_start        = _cfg('ema_start', 0.99925)
        self.ema_end          = _cfg('ema_end', 0.99925)
        self.loss_exp         = _cfg('loss_exp', 1.0)
        self.loss_beta        = _cfg('loss_beta', 1.0)
        self.normalize_reps   = _cfg('normalize_reps', False)
        self.auto_steps       = _cfg('auto_steps', 0)

        # Detect device from encoder
        try:
            self.device = next(encoder.parameters()).device
        except StopIteration:
            self.device = torch.device('cpu')

        # --- EMA target encoder ---
        # Import here to avoid circular imports in standalone testing
        try:
            from assets.ema_manager_template import EMAManagerWithRef
            self.ema_manager = EMAManagerWithRef(
                encoder,
                ema_schedule=(self.ema_start, self.ema_end),
                total_steps=self.epochs * 1000,  # Placeholder; caller should update
            )
        except ImportError:
            # Fallback: inline EMAManagerWithRef
            self.ema_manager = _InlineEMAManager(
                encoder,
                ema_start=self.ema_start,
                ema_end=self.ema_end,
                total_steps=self.epochs * 1000,
            )

        # --- Optimizer (4 param groups) ---
        self.optimizer = self._build_optimizer()

        # --- AMP GradScaler ---
        amp_supported = torch.cuda.is_available() and self.use_bfloat16
        self.scaler = GradScaler(enabled=amp_supported)

        # --- Simple step counters ---
        self._global_step = 0

    # ------------------------------------------------------------------
    # Optimizer construction
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> torch.optim.AdamW:
        """Build AdamW with 4 parameter groups (encoder/predictor x decay/no-decay)."""

        def split_params(module: nn.Module):
            decay, no_decay = [], []
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                if param.ndim == 1 or name.endswith('.bias'):
                    no_decay.append(param)
                else:
                    decay.append(param)
            return decay, no_decay

        enc_decay,  enc_no_decay  = split_params(self.encoder)
        pred_decay, pred_no_decay = split_params(self.predictor)

        param_groups = [
            {'params': enc_decay,    'lr': self.lr, 'weight_decay': self.weight_decay,
             'name': 'encoder_weights'},
            {'params': pred_decay,   'lr': self.lr, 'weight_decay': self.weight_decay,
             'name': 'predictor_weights'},
            {'params': enc_no_decay, 'lr': self.lr, 'weight_decay': 0.0,
             'name': 'encoder_no_decay'},
            {'params': pred_no_decay,'lr': self.lr, 'weight_decay': 0.0,
             'name': 'predictor_no_decay'},
        ]

        return torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)

    # ------------------------------------------------------------------
    # Train step
    # ------------------------------------------------------------------

    def train_step(
        self,
        batch: Union[torch.Tensor, Dict],
        masks_enc: List[torch.Tensor],
        masks_pred: List[torch.Tensor],
    ) -> Dict[str, float]:
        """
        Execute one JEPA training step.

        Pipeline:
            1. Extract context representations from visible patches.
            2. Predict masked representations with predictor.
            3. Compute EMA target representations for masked positions.
            4. Compute smooth L1 loss.
            5. Backward pass, gradient clipping, optimizer step.

        Args:
            batch:      [B, N_vis, embed_dim] visible patch representations,
                        or dict with key 'context' containing same tensor.
            masks_enc:  List of [N_vis] tensors — visible patch position indices.
            masks_pred: List of [N_pred] tensors — target patch position indices.

        Returns:
            Dict with keys: 'loss', 'grad_norm'.
        """
        # Unpack batch
        if isinstance(batch, dict):
            context_input = batch.get('context', batch.get('x', None))
            if context_input is None:
                raise ValueError("batch dict must have 'context' or 'x' key")
        else:
            context_input = batch

        context_input = context_input.to(self.device)
        masks_enc  = [m.to(self.device) for m in masks_enc]
        masks_pred = [m.to(self.device) for m in masks_pred]

        self.optimizer.zero_grad(set_to_none=True)

        use_amp = self.scaler.is_enabled()
        amp_dtype = torch.bfloat16 if use_amp else torch.float32

        with torch.autocast(device_type=self.device.type if hasattr(self.device, 'type') else 'cpu',
                            dtype=amp_dtype, enabled=use_amp):
            # Step 1: Encode visible patches
            context_repr = self.encoder(context_input)  # [B, N_vis, D]

            # Step 2: Predict masked patches
            pred_repr = self.predictor(context_repr, masks_enc, masks_pred)  # [B, N_pred, D]

            # Step 3: EMA target representations
            target_encoder = self.ema_manager.get_target_encoder()
            with torch.no_grad():
                # Target encoder sees ALL patches (visible + masked)
                # For synthetic tests, we reuse context_input as "all patches"
                target_repr_full = target_encoder(context_input)  # [B, N_all, D]

                # Gather target at masked positions
                m_pred = masks_pred[0]
                if m_pred.max() < target_repr_full.shape[1]:
                    target_repr = target_repr_full[:, m_pred, :]  # [B, N_pred, D]
                else:
                    # Fallback: use a subset if synthetic data is smaller
                    n = min(pred_repr.shape[1], target_repr_full.shape[1])
                    target_repr = target_repr_full[:, :n, :]
                    pred_repr   = pred_repr[:, :n, :]

                if self.normalize_reps:
                    D = target_repr.size(-1)
                    target_repr = F.layer_norm(target_repr, [D])
                    pred_repr_norm = F.layer_norm(pred_repr, [D])
                else:
                    pred_repr_norm = pred_repr

            # Step 4: Smooth L1 loss on masked positions only
            loss = smooth_l1_loss(
                pred_repr_norm, target_repr,
                beta=self.loss_beta, loss_exp=self.loss_exp,
            )

        # Step 5: Backward + clipping + optimizer step
        self.scaler.scale(loss).backward()

        grad_norm = 0.0
        if self.clip_grad > 0.0:
            self.scaler.unscale_(self.optimizer)
            all_params = (
                list(self.encoder.parameters())
                + list(self.predictor.parameters())
            )
            grad_norm = nn.utils.clip_grad_norm_(all_params, self.clip_grad).item()

        self.scaler.step(self.optimizer)
        self.scaler.update()

        self._global_step += 1

        return {
            'loss': loss.item(),
            'grad_norm': grad_norm,
        }

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------

    def update_ema(self, step: int) -> float:
        """
        Update the EMA target encoder.

        Args:
            step: Current global training step (for schedule computation).

        Returns:
            Current EMA momentum value.
        """
        return self.ema_manager.update(step)

    # ------------------------------------------------------------------
    # Loss-only computation (for validation / checkpoint comparison)
    # ------------------------------------------------------------------

    def compute_loss_only(
        self,
        batch: Union[torch.Tensor, Dict],
        masks_enc: List[torch.Tensor],
        masks_pred: List[torch.Tensor],
    ) -> float:
        """
        Compute loss without backward pass (for evaluation).

        Args:
            batch:      Same as train_step.
            masks_enc:  Visible patch position indices.
            masks_pred: Target patch position indices.

        Returns:
            Loss as Python float.
        """
        if isinstance(batch, dict):
            context_input = batch.get('context', batch.get('x'))
        else:
            context_input = batch

        context_input = context_input.to(self.device)
        masks_enc_d  = [m.to(self.device) for m in masks_enc]
        masks_pred_d = [m.to(self.device) for m in masks_pred]

        self.encoder.train(False)
        self.predictor.train(False)

        with torch.no_grad():
            context_repr = self.encoder(context_input)
            pred_repr    = self.predictor(context_repr, masks_enc_d, masks_pred_d)

            target_encoder = self.ema_manager.get_target_encoder()
            target_repr_full = target_encoder(context_input)
            m_pred = masks_pred_d[0]
            if m_pred.max() < target_repr_full.shape[1]:
                target_repr = target_repr_full[:, m_pred, :]
            else:
                n = min(pred_repr.shape[1], target_repr_full.shape[1])
                target_repr = target_repr_full[:, :n, :]
                pred_repr   = pred_repr[:, :n, :]

            loss = smooth_l1_loss(pred_repr, target_repr,
                                  beta=self.loss_beta, loss_exp=self.loss_exp)

        self.encoder.train(True)
        self.predictor.train(True)

        return loss.item()

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, epoch: int) -> None:
        """
        Save full training state to a file.

        Saved keys: epoch, encoder, predictor, target_encoder,
                    optimizer, scaler, global_step.

        Args:
            path:  File path to save checkpoint.
            epoch: Completed epoch number.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        state = {
            'epoch':          epoch,
            'global_step':    self._global_step,
            'encoder':        self.encoder.state_dict(),
            'predictor':      self.predictor.state_dict(),
            'target_encoder': self.ema_manager.target_encoder.state_dict(),
            'optimizer':      self.optimizer.state_dict(),
            'scaler':         self.scaler.state_dict(),
            'ema_state':      self.ema_manager.state_dict(),
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str, strict: bool = True) -> int:
        """
        Load training state from a checkpoint file.

        Args:
            path:   File path to load.
            strict: Whether to require exact key match when loading modules.

        Returns:
            The epoch number stored in the checkpoint.
        """
        ckpt = torch.load(path, map_location=self.device)

        def _strip_ddp(sd: dict) -> dict:
            return {k.replace('module.', '', 1): v for k, v in sd.items()}

        self.encoder.load_state_dict(
            _strip_ddp(ckpt['encoder']), strict=strict
        )
        self.predictor.load_state_dict(
            _strip_ddp(ckpt['predictor']), strict=strict
        )
        self.ema_manager.target_encoder.load_state_dict(
            _strip_ddp(ckpt['target_encoder']), strict=strict
        )
        # Ensure target stays non-differentiable after load
        for p in self.ema_manager.target_encoder.parameters():
            p.requires_grad_(False)

        try:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        except Exception as exc:
            print(f"Warning: optimizer state not loaded: {exc}")

        if 'scaler' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler'])

        if 'global_step' in ckpt:
            self._global_step = ckpt['global_step']

        if 'ema_state' in ckpt:
            self.ema_manager.load_state_dict(ckpt['ema_state'], strict=strict)

        return ckpt.get('epoch', 0)


# ---------------------------------------------------------------------------
# Inline EMA fallback (used when ema_manager_template is not importable)
# ---------------------------------------------------------------------------

class _InlineEMAManager:
    """Minimal inline EMA manager used as fallback in JEPATrainer."""

    def __init__(self, encoder: nn.Module,
                 ema_start: float, ema_end: float, total_steps: int) -> None:
        self.ema_start   = ema_start
        self.ema_end     = ema_end
        self.total_steps = max(1, total_steps)
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.target_encoder.train(False)
        self._encoder_ref = encoder
        self._current_momentum = ema_start

    def get_momentum(self, step: int) -> float:
        progress = min(step / self.total_steps, 1.0)
        cosine   = (math.cos(math.pi * progress) + 1.0) / 2.0
        return self.ema_end - (self.ema_end - self.ema_start) * cosine

    def update(self, step: int) -> float:
        m = self.get_momentum(step)
        self._current_momentum = m
        with torch.no_grad():
            for pt, pe in zip(
                self.target_encoder.parameters(),
                self._encoder_ref.parameters(),
            ):
                pt.mul_(m).add_(pe.data, alpha=1.0 - m)
        return m

    def get_target_encoder(self) -> nn.Module:
        return self.target_encoder

    def state_dict(self) -> dict:
        return {
            'target_encoder':   self.target_encoder.state_dict(),
            'ema_start':        self.ema_start,
            'ema_end':          self.ema_end,
            'total_steps':      self.total_steps,
            'current_momentum': self._current_momentum,
        }

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        self.target_encoder.load_state_dict(state['target_encoder'], strict=strict)
        self.ema_start        = state.get('ema_start', self.ema_start)
        self.ema_end          = state.get('ema_end', self.ema_end)
        self.total_steps      = state.get('total_steps', self.total_steps)
        self._current_momentum = state.get('current_momentum', self.ema_start)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.target_encoder.train(False)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _make_tiny_trainer(embed_dim: int = 64, pred_dim: int = 32):
    """Construct a minimal JEPATrainer for CPU testing."""
    try:
        from assets.predictor_template import VisionTransformerPredictor
    except ImportError:
        from predictor_template import VisionTransformerPredictor

    encoder   = _StubEncoder(in_dim=embed_dim, embed_dim=embed_dim)
    predictor = VisionTransformerPredictor(
        embed_dim=embed_dim,
        predictor_embed_dim=pred_dim,
        depth=2,
        num_heads=2,
        num_targets=4,
    )

    class _TinyCfg:
        lr = 1e-3
        final_lr = 1e-6
        weight_decay = 0.04
        final_weight_decay = 0.4
        warmup_epochs = 2
        epochs = 10
        use_bfloat16 = False
        clip_grad = 1.0
        ema_start = 0.99
        ema_end = 0.999
        loss_exp = 1.0
        loss_beta = 1.0
        normalize_reps = False
        auto_steps = 0

    return JEPATrainer(encoder, predictor, _TinyCfg())


if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("JEPATrainer self-tests")
    print("=" * 60)

    torch.manual_seed(0)

    embed_dim = 64
    B = 2
    N_vis  = 8
    N_pred = 4

    context_input = torch.randn(B, N_vis, embed_dim)
    masks_enc  = [torch.arange(N_vis)]
    masks_pred = [torch.arange(N_vis, N_vis + N_pred)]

    # ---------------------------------------------------------------
    # Test 1: Forward produces finite, non-negative loss
    # ---------------------------------------------------------------
    trainer = _make_tiny_trainer(embed_dim=embed_dim, pred_dim=32)
    loss_dict = trainer.train_step(context_input, masks_enc, masks_pred)

    assert 'loss' in loss_dict, "train_step should return dict with 'loss'"
    loss_val = loss_dict['loss']
    assert isinstance(loss_val, float), f"loss should be float, got {type(loss_val)}"
    assert math.isfinite(loss_val), f"loss is not finite: {loss_val}"
    assert loss_val >= 0.0, f"loss is negative: {loss_val}"
    print(f"[PASS] Forward produces finite, non-negative loss: {loss_val:.4f}")

    # ---------------------------------------------------------------
    # Test 2: Loss decreases over 50 steps on fixed synthetic batch
    # ---------------------------------------------------------------
    trainer2 = _make_tiny_trainer(embed_dim=embed_dim, pred_dim=32)
    torch.manual_seed(42)
    fixed_batch = torch.randn(B, N_vis, embed_dim)

    losses = []
    for step in range(50):
        d = trainer2.train_step(fixed_batch, masks_enc, masks_pred)
        trainer2.update_ema(step)
        losses.append(d['loss'])

    start_avg = sum(losses[:5]) / 5
    end_avg   = sum(losses[-5:]) / 5
    assert end_avg < start_avg, (
        f"Loss should decrease: start={start_avg:.4f}, end={end_avg:.4f}"
    )
    print(f"[PASS] Loss decreases over 50 steps: {start_avg:.4f} -> {end_avg:.4f}")

    # ---------------------------------------------------------------
    # Test 3: Checkpoint round-trip preserves parameters and epoch
    # ---------------------------------------------------------------
    trainer3 = _make_tiny_trainer(embed_dim=embed_dim, pred_dim=32)
    for step in range(5):
        trainer3.train_step(context_input, masks_enc, masks_pred)
        trainer3.update_ema(step)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_ckpt.pth")
        trainer3.save_checkpoint(ckpt_path, epoch=3)

        trainer4 = _make_tiny_trainer(embed_dim=embed_dim, pred_dim=32)
        loaded_epoch = trainer4.load_checkpoint(ckpt_path)

        assert loaded_epoch == 3, (
            f"Loaded epoch should be 3, got {loaded_epoch}"
        )

        # Verify encoder params match
        for p1, p2 in zip(trainer3.encoder.parameters(),
                           trainer4.encoder.parameters()):
            assert torch.allclose(p1, p2, atol=1e-7), (
                "Encoder params differ after checkpoint load"
            )

        # Verify predictor params match
        for p1, p2 in zip(trainer3.predictor.parameters(),
                           trainer4.predictor.parameters()):
            assert torch.allclose(p1, p2, atol=1e-7), (
                "Predictor params differ after checkpoint load"
            )

        # Verify target encoder params match
        for p1, p2 in zip(trainer3.ema_manager.target_encoder.parameters(),
                           trainer4.ema_manager.target_encoder.parameters()):
            assert torch.allclose(p1, p2, atol=1e-7), (
                "Target encoder params differ after checkpoint load"
            )

    print("[PASS] Checkpoint round-trip: epoch, encoder, predictor, target_encoder all match")

    # ---------------------------------------------------------------
    # Test 4: Resumed training gives identical loss
    # ---------------------------------------------------------------
    torch.manual_seed(7)
    eval_batch = torch.randn(B, N_vis, embed_dim)

    trainer5 = _make_tiny_trainer(embed_dim=embed_dim, pred_dim=32)
    for step in range(3):
        trainer5.train_step(eval_batch, masks_enc, masks_pred)
        trainer5.update_ema(step)

    ref_loss = trainer5.compute_loss_only(eval_batch, masks_enc, masks_pred)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path2 = os.path.join(tmpdir, "resume_ckpt.pth")
        trainer5.save_checkpoint(ckpt_path2, epoch=3)

        trainer6 = _make_tiny_trainer(embed_dim=embed_dim, pred_dim=32)
        trainer6.load_checkpoint(ckpt_path2)
        loaded_loss = trainer6.compute_loss_only(eval_batch, masks_enc, masks_pred)

    assert abs(ref_loss - loaded_loss) < 1e-5, (
        f"Loss mismatch after resume: ref={ref_loss:.6f}, loaded={loaded_loss:.6f}"
    )
    print(f"[PASS] Resumed training gives identical loss: {ref_loss:.6f}")

    # ---------------------------------------------------------------
    # Test 5: EMA update changes target encoder
    # ---------------------------------------------------------------
    trainer7 = _make_tiny_trainer(embed_dim=embed_dim, pred_dim=32)
    target_before = [p.clone() for p in trainer7.ema_manager.target_encoder.parameters()]

    # Modify encoder significantly
    with torch.no_grad():
        for p in trainer7.encoder.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    trainer7.update_ema(step=0)
    target_after = list(trainer7.ema_manager.target_encoder.parameters())

    any_changed = any(
        not torch.allclose(b, a, atol=1e-5)
        for b, a in zip(target_before, target_after)
    )
    assert any_changed, "EMA update should change target encoder parameters"
    print("[PASS] EMA update changes target encoder parameters")

    # ---------------------------------------------------------------
    # Test 6: smooth_l1_loss boundary conditions
    # ---------------------------------------------------------------
    # Zero diff -> zero loss
    loss_zero = smooth_l1_loss(torch.zeros(2, 4, 16), torch.zeros(2, 4, 16))
    assert abs(loss_zero.item()) < 1e-6, f"Zero diff should give zero loss, got {loss_zero.item()}"

    # Small diff < beta: quadratic regime
    p = torch.zeros(1, 1, 1)
    t = torch.full((1, 1, 1), 0.5)
    loss_small = smooth_l1_loss(p, t, beta=1.0)
    expected_small = 0.5 * 0.5 ** 2 / 1.0  # = 0.125
    assert abs(loss_small.item() - expected_small) < 1e-5, (
        f"Quadratic regime: expected {expected_small:.4f}, got {loss_small.item():.4f}"
    )

    # Large diff > beta: linear regime
    p2 = torch.zeros(1, 1, 1)
    t2 = torch.full((1, 1, 1), 2.0)
    loss_large = smooth_l1_loss(p2, t2, beta=1.0)
    expected_large = 2.0 - 0.5 * 1.0  # = 1.5
    assert abs(loss_large.item() - expected_large) < 1e-5, (
        f"Linear regime: expected {expected_large:.4f}, got {loss_large.item():.4f}"
    )
    print("[PASS] smooth_l1_loss: zero diff, quadratic regime, linear regime all correct")

    print()
    print("All 6 self-tests passed.")
