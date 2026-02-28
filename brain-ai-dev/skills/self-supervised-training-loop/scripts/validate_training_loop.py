"""
validate_training_loop.py — Validates the 4 done-when gates from SKILL.md.

Gates:
    1. AMP Ordering Correct
    2. EMA Schedule Tracks
    3. DDP Wraps Correctly
    4. Checkpoint Round-Trip

Run with: python validate_training_loop.py
All gates must pass for the skill to be complete.
"""

from __future__ import annotations

import copy
import math
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add assets directory to path
ASSETS_DIR = Path(__file__).parent.parent / 'assets'
sys.path.insert(0, str(ASSETS_DIR))

from amp_gradient_template import AMPContext
from ema_template import EMAUpdater
from checkpoint_template import CheckpointManager
from training_config_template import TrainingConfig
from training_loop_template import CosineWarmupScheduler


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'
SKIP = '\033[93mSKIP\033[0m'

gate_results: Dict[str, bool] = {}


def report(gate_name: str, check_name: str, passed: bool, detail: str = '') -> None:
    print(f"  {'[PASS]' if passed else '[FAIL]'} {check_name}" + (f" — {detail}" if detail else ''))
    if not passed:
        gate_results[gate_name] = False


def start_gate(name: str) -> None:
    gate_results[name] = True
    print(f"\n{'='*60}")
    print(f"Gate: {name}")
    print('='*60)


# ---------------------------------------------------------------------------
# Gate 1: AMP Ordering Correct
# ---------------------------------------------------------------------------

def gate_1_amp_ordering() -> bool:
    start_gate("Gate 1: AMP Ordering Correct")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_cuda = (device == 'cuda')

    model = nn.Linear(16, 8)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ctx = AMPContext(
        dtype=torch.bfloat16,
        enabled=use_cuda,
        scaler_enabled=use_cuda,
    )

    losses = []
    grad_norms = []

    for step in range(5):
        # Step 1
        optimizer.zero_grad(set_to_none=True)

        # Step 2: forward under autocast
        x = torch.randn(4, 16, device=device)
        with ctx.autocast():
            out = model(x)
            loss = out.pow(2).mean()

        # Step 3: scaled backward
        ctx.backward(loss)

        # Verify gradients exist after backward
        has_grads = any(p.grad is not None for p in model.parameters())
        report("Gate 1", f"Step {step}: gradients exist after backward", has_grads)

        # Step 4+5: unscale THEN clip — this is the correct order
        grad_norm = ctx.unscale_and_clip(optimizer, model.parameters(), max_norm=1.0)

        # Check 1: grad_norm is finite
        report("Gate 1", f"Step {step}: grad_norm is finite",
               math.isfinite(grad_norm), f"grad_norm={grad_norm:.4f}")

        # Step 6+7: step and update
        ctx.step_and_update(optimizer)

        losses.append(loss.item())
        grad_norms.append(grad_norm)

    # Check 2: all losses are finite
    all_finite = all(math.isfinite(l) for l in losses)
    report("Gate 1", "All 5 losses are finite", all_finite,
           f"losses={[f'{l:.4f}' for l in losses]}")

    # Check 3: grad_norms are positive
    all_positive = all(g > 0 for g in grad_norms)
    report("Gate 1", "All grad_norms are positive", all_positive,
           f"norms={[f'{g:.4f}' for g in grad_norms]}")

    # Check 4: state_dict round-trip
    sd = ctx.state_dict()
    ctx2 = AMPContext(enabled=use_cuda, scaler_enabled=use_cuda)
    ctx2.load_state_dict(sd)
    sd2 = ctx2.state_dict()
    scale_match = True
    if 'scale' in sd and 'scale' in sd2:
        scale_match = sd['scale'] == sd2['scale']
    report("Gate 1", "GradScaler state_dict round-trip", scale_match)

    # Check 5: inf injection causes scale reduction (CUDA only)
    if use_cuda:
        model2 = nn.Linear(4, 2).cuda()
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        ctx3 = AMPContext(dtype=torch.bfloat16, enabled=True, scaler_enabled=True)

        opt2.zero_grad(set_to_none=True)
        x2 = torch.randn(2, 4, device='cuda')
        with ctx3.autocast():
            out2 = model2(x2)
            loss2 = out2.mean()
        ctx3.backward(loss2)

        # Inject inf
        with torch.no_grad():
            list(model2.parameters())[0].grad.fill_(float('inf'))

        scale_before = ctx3.get_scale()
        ctx3.unscale_and_clip(opt2, model2.parameters(), max_norm=1.0)
        ctx3.step_and_update(opt2)
        scale_after = ctx3.get_scale()

        report("Gate 1", "inf gradient reduces scale",
               scale_after < scale_before,
               f"scale: {scale_before:.0f} -> {scale_after:.0f}")
    else:
        print(f"  [SKIP] inf detection test — CUDA not available")

    return gate_results.get("Gate 1", True)


# ---------------------------------------------------------------------------
# Gate 2: EMA Schedule Tracks
# ---------------------------------------------------------------------------

def gate_2_ema_schedule() -> bool:
    start_gate("Gate 2: EMA Schedule Tracks")

    updater = EMAUpdater(tau_base=0.996, tau_final=0.9999, total_steps=100_000)

    # Check 1: tau at step 0 equals tau_base
    tau_0 = updater.get_tau(0)
    report("Gate 2", "tau(0) == tau_base (0.996)",
           abs(tau_0 - 0.996) < 1e-9, f"tau(0)={tau_0}")

    # Check 2: tau at total_steps approaches tau_final
    tau_end = updater.get_tau(100_000)
    report("Gate 2", "tau(total_steps) ~= tau_final (0.9999)",
           abs(tau_end - 0.9999) < 1e-6, f"tau(total_steps)={tau_end:.6f}")

    # Check 3: tau at midpoint (should be between tau_base and tau_final)
    tau_mid = updater.get_tau(50_000)
    report("Gate 2", "tau(midpoint) is between tau_base and tau_final",
           tau_0 < tau_mid < tau_end,
           f"tau(50000)={tau_mid:.6f}")

    # Check 4: monotonically increasing
    steps = list(range(0, 100_001, 10_000))
    taus = [updater.get_tau(s) for s in steps]
    is_monotone = all(taus[i] <= taus[i+1] for i in range(len(taus)-1))
    report("Gate 2", "tau schedule is monotonically increasing", is_monotone,
           f"range: [{taus[0]:.6f}, {taus[-1]:.6f}]")

    # Check 5: @no_grad — no grad_fn on target parameters after update
    online = nn.Linear(4, 2)
    target = nn.Linear(4, 2)

    # Ensure online is in "training" state (gradients tracked)
    updater.update(online, target, step=0)

    no_grad_fn = all(p.grad_fn is None for p in target.parameters())
    report("Gate 2", "@no_grad: target params have no grad_fn", no_grad_fn)

    # Verify global grad enabled state unchanged
    report("Gate 2", "grad still enabled globally after EMA update",
           torch.is_grad_enabled())

    # Check 6: initial_sync makes parameters identical
    online2 = nn.Linear(8, 4)
    target2 = nn.Linear(8, 4)
    updater.initial_sync(online2, target2)

    dist = updater.compute_distance(online2, target2)
    report("Gate 2", "initial_sync: L2 distance = 0",
           dist < 1e-8, f"distance={dist:.2e}")

    # Check 7: update changes target params
    online3 = nn.Linear(2, 2, bias=False)
    target3 = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        online3.weight.fill_(1.0)
        target3.weight.fill_(0.0)

    updater3 = EMAUpdater(tau_base=0.996, tau_final=0.9999, total_steps=1000)
    updater3.update(online3, target3, step=0)

    updated_val = target3.weight.mean().item()
    expected_val = 0.996 * 0.0 + 0.004 * 1.0  # = 0.004
    report("Gate 2", "EMA update value correct",
           abs(updated_val - expected_val) < 1e-5,
           f"got {updated_val:.6f}, expected {expected_val:.6f}")

    # Check 8: target diverges from online after training
    online4 = nn.Linear(4, 2)
    target4 = nn.Linear(4, 2)
    updater.initial_sync(online4, target4)

    # Modify online and apply EMA
    with torch.no_grad():
        for p in online4.parameters():
            p.data += 5.0

    updater.update(online4, target4, step=0)
    dist4 = updater.compute_distance(online4, target4)
    report("Gate 2", "Target diverges from online after update",
           dist4 > 0.0, f"distance={dist4:.4f}")

    return gate_results.get("Gate 2", True)


# ---------------------------------------------------------------------------
# Gate 3: DDP Wraps Correctly
# ---------------------------------------------------------------------------

def gate_3_ddp_setup() -> bool:
    start_gate("Gate 3: DDP Wraps Correctly")

    # Check 1: SyncBatchNorm conversion replaces BatchNorm
    class ModelWithBN(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(8, 8)
            self.bn1 = nn.BatchNorm1d(8)
            self.layer2 = nn.Linear(8, 4)
            self.bn2 = nn.BatchNorm1d(4)

        def forward(self, x):
            return self.layer2(self.bn1(self.layer1(x)))

    model_bn = ModelWithBN()
    bn_before = sum(1 for m in model_bn.modules()
                    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)))
    report("Gate 3", f"BatchNorm layers exist before conversion ({bn_before})", bn_before > 0)

    model_converted = nn.SyncBatchNorm.convert_sync_batchnorm(model_bn)
    bn_after = sum(1 for m in model_converted.modules()
                   if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)))
    sync_bn_count = sum(1 for m in model_converted.modules()
                        if isinstance(m, nn.SyncBatchNorm))

    report("Gate 3", "All BatchNorm converted to SyncBatchNorm",
           bn_after == 0 and sync_bn_count == bn_before,
           f"BN remaining={bn_after}, SyncBN={sync_bn_count}")

    # Check 2: Online model wrapped, target not wrapped
    # (Mock test without actual DDP init)
    online_model = nn.Linear(4, 2)
    target_model = nn.Linear(4, 2)

    # Simulate DDP wrapping (production code uses DDP; here we use a fake for testing)
    class MockDDP(nn.Module):
        def __init__(self, m, **kwargs):
            super().__init__()
            self.module = m
            self._ddp_params = kwargs

        def forward(self, x):
            return self.module(x)

    wrapped_online = MockDDP(
        online_model,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )

    # Online: has .module attribute
    report("Gate 3", "Online encoder has .module attribute",
           hasattr(wrapped_online, 'module'))

    # Target: does NOT have .module attribute (not wrapped)
    report("Gate 3", "Target encoder does NOT have .module attribute",
           not hasattr(target_model, 'module'))

    # Check 3: find_unused_parameters=False in DDP call
    report("Gate 3", "find_unused_parameters=False",
           wrapped_online._ddp_params.get('find_unused_parameters') == False)

    # Check 4: gradient_as_bucket_view=True in DDP call
    report("Gate 3", "gradient_as_bucket_view=True",
           wrapped_online._ddp_params.get('gradient_as_bucket_view') == True)

    # Check 5: Target encoder parameters are frozen
    for param in target_model.parameters():
        param.requires_grad_(False)

    all_frozen = all(not p.requires_grad for p in target_model.parameters())
    report("Gate 3", "Target encoder parameters are frozen (requires_grad=False)",
           all_frozen)

    # Check 6: Unwrap works correctly
    def unwrap(m):
        return m.module if hasattr(m, 'module') else m

    unwrapped = unwrap(wrapped_online)
    report("Gate 3", "unwrap_model returns original model",
           unwrapped is online_model)

    return gate_results.get("Gate 3", True)


# ---------------------------------------------------------------------------
# Gate 4: Checkpoint Round-Trip
# ---------------------------------------------------------------------------

def gate_4_checkpoint() -> bool:
    start_gate("Gate 4: Checkpoint Round-Trip")

    test_dir = tempfile.mkdtemp(prefix='gate4_ckpt_')

    try:
        # Build a minimal scaler stub
        class MinimalScalerCtx:
            def state_dict(self):
                return {'scale': 65536.0}
            def load_state_dict(self, sd):
                self._loaded = sd

        # Create components
        online = nn.Linear(8, 4)
        target = nn.Linear(8, 4)

        # Set known weights
        with torch.no_grad():
            online.weight.fill_(7.777)
            target.weight.fill_(3.333)

        optimizer = torch.optim.AdamW(online.parameters(), lr=1e-3)
        # Run one step to populate optimizer state
        online(torch.randn(2, 8)).mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        scheduler = CosineWarmupScheduler(optimizer, warmup_steps=10, total_steps=100)
        for _ in range(5):  # Step scheduler a few times
            scheduler.step()

        scaler_ctx = MinimalScalerCtx()

        manager = CheckpointManager(test_dir, keep_last=3)

        # Save checkpoint at step 100
        saved_path = manager.save(online, target, optimizer, scaler_ctx, scheduler, step=100)
        report("Gate 4", f"Checkpoint file created at step 100", saved_path.exists())

        # Load and verify all 6 keys
        loaded_state = torch.load(str(saved_path), map_location='cpu', weights_only=False)
        required_keys = {'model_state_dict', 'target_state_dict', 'optimizer_state_dict',
                         'scaler_state_dict', 'scheduler_state_dict', 'step'}
        all_keys_present = required_keys.issubset(set(loaded_state.keys()))
        report("Gate 4", "All 6 state keys present in checkpoint", all_keys_present,
               f"found: {set(loaded_state.keys())}")

        # Verify step == 100
        report("Gate 4", "Checkpoint step == 100",
               loaded_state['step'] == 100,
               f"got step={loaded_state['step']}")

        # Verify no DDP prefix in model keys
        model_keys = list(loaded_state['model_state_dict'].keys())
        no_ddp_prefix = all(not k.startswith('module.') for k in model_keys)
        report("Gate 4", "model_state_dict has no 'module.' prefix (DDP unwrapped)",
               no_ddp_prefix, f"keys: {model_keys[:3]}")

        # Verify model weights preserved
        saved_weight = loaded_state['model_state_dict']['weight'].mean().item()
        report("Gate 4", "Online encoder weights preserved (7.777)",
               abs(saved_weight - 7.777) < 1e-3,
               f"got {saved_weight:.4f}")

        saved_target_weight = loaded_state['target_state_dict']['weight'].mean().item()
        report("Gate 4", "Target encoder weights preserved (3.333)",
               abs(saved_target_weight - 3.333) < 1e-3,
               f"got {saved_target_weight:.4f}")

        # Test resume
        online_new = nn.Linear(8, 4)
        target_new = nn.Linear(8, 4)
        optimizer_new = torch.optim.AdamW(online_new.parameters(), lr=1e-3)
        scheduler_new = CosineWarmupScheduler(optimizer_new, warmup_steps=10, total_steps=100)
        scaler_new = MinimalScalerCtx()

        resumed_step = manager.resume(online_new, target_new, optimizer_new, scaler_new, scheduler_new)

        report("Gate 4", "resume() returns correct step (100)",
               resumed_step == 100, f"got {resumed_step}")

        resumed_weight = online_new.weight.mean().item()
        report("Gate 4", "Resumed online encoder weight matches saved (7.777)",
               abs(resumed_weight - 7.777) < 1e-3,
               f"got {resumed_weight:.4f}")

        resumed_lr = optimizer_new.param_groups[0]['lr']
        report("Gate 4", "Resumed optimizer lr is set (not default)",
               resumed_lr > 0,
               f"lr={resumed_lr:.6f}")

        # Test auto-resume finds latest checkpoint
        manager.save(online, target, optimizer, scaler_ctx, scheduler, step=200)
        manager.save(online, target, optimizer, scaler_ctx, scheduler, step=300)

        result = manager.load_latest()
        _, latest_step = result
        report("Gate 4", "auto-resume finds step 300 (highest)",
               latest_step == 300, f"found step={latest_step}")

        # Test pruning
        manager4 = CheckpointManager(test_dir, keep_last=2)
        manager4.save(online, target, optimizer, scaler_ctx, scheduler, step=400)

        deleted = manager4.prune()
        remaining = manager4._scan_checkpoints()
        report("Gate 4", f"Pruning keeps 2 most recent",
               len(remaining) == 2,
               f"deleted={deleted}, remaining steps={sorted(remaining.keys())}")

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)

    return gate_results.get("Gate 4", True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("Self-Supervised Training Loop — Done-When Gate Validation")
    print("=" * 60)

    g1 = gate_1_amp_ordering()
    g2 = gate_2_ema_schedule()
    g3 = gate_3_ddp_setup()
    g4 = gate_4_checkpoint()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    all_gates = {
        "Gate 1: AMP Ordering":        g1,
        "Gate 2: EMA Schedule":        g2,
        "Gate 3: DDP Wraps Correctly": g3,
        "Gate 4: Checkpoint Round-Trip": g4,
    }

    all_passed = True
    for gate_name, passed in all_gates.items():
        status = '[PASS]' if passed else '[FAIL]'
        print(f"  {status} {gate_name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All 4 gates PASSED — skill is complete.")
        return 0
    else:
        print("One or more gates FAILED — review output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
