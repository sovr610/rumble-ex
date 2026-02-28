#!/usr/bin/env python3
"""
validate_pipeline.py — Validates done-when gates for the DataLoader Throughput
+ Sequence Packing skill.

Gates:
  1. Pipeline Measured — PipelineAuditor produces valid metrics with stall ratio
  2. Streaming Works — At least one backend loads data sequentially
  3. Packing Eliminates Padding — padding_ratio drops, cu_seqlens/position_ids correct
  4. Deterministic Sharding — Different ranks get different data, same seed+epoch = same order

Usage:
    python scripts/validate_pipeline.py [--gate 1|2|3|4] [--all]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tempfile
from pathlib import Path

# Add parent directories to path
SKILL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_DIR / "assets"))

import numpy as np


def validate_gate_1() -> bool:
    """Gate 1: Pipeline Measured — PipelineAuditor produces valid metrics."""
    print("\n" + "=" * 60)
    print("Gate 1: Pipeline Measured")
    print("=" * 60)

    from pipeline_auditor_template import PipelineAuditor, PipelineMetrics

    errors = []

    # Create auditor and simulate a training loop
    auditor = PipelineAuditor(device=None)
    for step in range(20):
        auditor.mark_data_start()
        time.sleep(0.002)  # simulate data loading
        auditor.mark_data_end()
        auditor.mark_compute_start()
        time.sleep(0.005)  # simulate compute
        auditor.mark_compute_end()
        auditor.record_tokens(raw=2048, effective=1800)

    auditor.padding_ratio = 0.12
    auditor.dataloader_settings = {
        "num_workers": 4, "prefetch_factor": 2,
        "persistent_workers": True, "pin_memory": True,
    }

    metrics = auditor.report()

    # Check required conditions
    if metrics.steps_measured != 20:
        errors.append(f"steps_measured={metrics.steps_measured}, expected 20")

    if not (0.0 <= metrics.data_stall_ratio_p50 <= 1.0):
        errors.append(f"data_stall_ratio_p50={metrics.data_stall_ratio_p50} out of [0,1]")

    if not (0.0 <= metrics.data_stall_ratio_p90 <= 1.0):
        errors.append(f"data_stall_ratio_p90={metrics.data_stall_ratio_p90} out of [0,1]")

    if metrics.data_stall_ratio_p90 < metrics.data_stall_ratio_p50 - 1e-6:
        errors.append(f"p90 < p50: {metrics.data_stall_ratio_p90} < {metrics.data_stall_ratio_p50}")

    if metrics.t_data_p50_ms <= 0:
        errors.append(f"t_data_p50_ms={metrics.t_data_p50_ms} should be > 0")

    # Check JSON output
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        auditor.to_json(tmp_path)
        with open(tmp_path) as f:
            data = json.load(f)
        required_keys = [
            "version", "steps_measured", "data_stall_ratio_p50",
            "data_stall_ratio_p90", "t_data_p50_ms", "padding_ratio",
        ]
        for key in required_keys:
            if key not in data:
                errors.append(f"JSON missing required key: {key}")
    finally:
        os.unlink(tmp_path)

    _print_gate_result("Gate 1", errors)
    return len(errors) == 0


def validate_gate_2() -> bool:
    """Gate 2: Streaming Works — At least one backend loads data."""
    print("\n" + "=" * 60)
    print("Gate 2: Streaming Works")
    print("=" * 60)

    from streaming_dataset_template import ShardedStreamDataset, StreamConfig, ShardCache

    errors = []

    # Test synthetic backend
    cfg = StreamConfig(format="synthetic", num_samples=50, target_seq_len=64)
    ds = ShardedStreamDataset(cfg, rank=0, world_size=1)
    batches = []
    for i, batch in enumerate(ds):
        batches.append(batch)
        if i >= 49:
            break

    if len(batches) == 0:
        errors.append("Synthetic backend yielded no batches")
    elif "input_ids" not in batches[0]:
        errors.append("Batch missing 'input_ids' key")
    elif len(batches[0]["input_ids"]) != 64:
        errors.append(f"input_ids length={len(batches[0]['input_ids'])}, expected 64")

    # Test memmap backend with temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = os.path.join(tmpdir, "test.bin")
        tokens = np.arange(0, 640, dtype=np.uint16)
        tokens.tofile(bin_path)

        mm_cfg = StreamConfig(
            format="memmap",
            data_path=os.path.join(tmpdir, "test"),
            target_seq_len=64,
        )
        ds_mm = ShardedStreamDataset(mm_cfg, rank=0, world_size=1)
        mm_batches = list(ds_mm)
        if len(mm_batches) == 0:
            errors.append("Memmap backend yielded no batches")

    # Test caching
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = ShardCache(tmpdir, max_bytes=10000)
        test_data = b"shard content for testing"
        cache.put("test_shard", test_data)
        retrieved = cache.get("test_shard")
        if retrieved != test_data:
            errors.append("Cache store/retrieve failed")

        # Checksum validation
        cache_path = Path(tmpdir) / "test_shard"
        cache_path.write_bytes(b"corrupted")
        if cache.get("test_shard") is not None:
            errors.append("Cache did not detect corruption")

    _print_gate_result("Gate 2", errors)
    return len(errors) == 0


def validate_gate_3() -> bool:
    """Gate 3: Packing Eliminates Padding."""
    print("\n" + "=" * 60)
    print("Gate 3: Packing Eliminates Padding")
    print("=" * 60)

    from sequence_packer_template import (
        SequencePacker, PackingConfig, TokenizedExample,
    )
    from packing_collators_template import PaddingCollator, PackingCollator

    errors = []

    # Create examples with varying lengths
    examples = [
        TokenizedExample(
            input_ids=list(range(length)),
            labels=list(range(length)),
        )
        for length in [30, 80, 50, 120, 45, 90, 60, 100]
    ]

    # SFT packing
    cfg = PackingConfig(
        mode="sft_boundary_aware",
        target_seq_len=512,
        bucket_boundaries=(64, 128, 256),
    )
    packer = SequencePacker(cfg)
    packed = packer.pack_sft(examples)

    # Check cu_seqlens
    if packed.cu_seqlens[0] != 0:
        errors.append(f"cu_seqlens[0]={packed.cu_seqlens[0]}, expected 0")

    if packed.cu_seqlens[-1] != len(packed.input_ids):
        errors.append(
            f"cu_seqlens[-1]={packed.cu_seqlens[-1]} != "
            f"len(input_ids)={len(packed.input_ids)}"
        )

    # Monotonicity
    for i in range(len(packed.cu_seqlens) - 1):
        if packed.cu_seqlens[i] >= packed.cu_seqlens[i + 1]:
            errors.append(f"cu_seqlens not monotonic at index {i}")
            break

    # position_ids reset at boundaries
    for i in range(len(packed.cu_seqlens) - 1):
        pos = packed.position_ids[packed.cu_seqlens[i]]
        if pos != 0:
            errors.append(f"position_ids[{packed.cu_seqlens[i]}]={pos}, expected 0")
            break

    # Padding ratio comparison: packing vs padding
    dict_examples = [
        {"input_ids": ex.input_ids, "labels": ex.labels}
        for ex in examples
    ]

    padding_collator = PaddingCollator(pad_token_id=0)
    padded_batch = padding_collator(dict_examples)
    padded_ratio = padding_collator.last_metrics.padding_ratio

    packed_ratio = packer.padding_ratio(packed)

    if packed_ratio >= padded_ratio:
        errors.append(
            f"Packing ratio ({packed_ratio:.4f}) should be less than "
            f"padding ratio ({padded_ratio:.4f})"
        )
    else:
        print(f"  Padding ratio: {padded_ratio:.4f} -> Packing ratio: {packed_ratio:.4f}")

    # Pretrain blocks
    pretrain_cfg = PackingConfig(
        mode="pretrain_blocks",
        target_seq_len=100,
    )
    pretrain_packer = SequencePacker(pretrain_cfg)
    docs = [[1] * 50, [2] * 80, [3] * 120, [4] * 200]
    blocks = list(pretrain_packer.pack_pretrain(iter(docs)))
    for i, block in enumerate(blocks):
        if len(block.input_ids) != 100:
            errors.append(f"Block {i} length={len(block.input_ids)}, expected 100")
            break

    block_ratio = pretrain_packer.padding_ratio_blocks(blocks)
    if block_ratio > 0.1:
        errors.append(f"Pretrain block padding_ratio={block_ratio:.4f}, expected < 0.1")

    _print_gate_result("Gate 3", errors)
    return len(errors) == 0


def validate_gate_4() -> bool:
    """Gate 4: Deterministic Sharding — ranks get different data, reproducible."""
    print("\n" + "=" * 60)
    print("Gate 4: Deterministic Sharding")
    print("=" * 60)

    from streaming_dataset_template import ShardedStreamDataset, StreamConfig

    errors = []

    cfg = StreamConfig(format="synthetic", num_samples=200, target_seq_len=32)

    # Different ranks get different data
    ds0 = ShardedStreamDataset(cfg, rank=0, world_size=4)
    ds1 = ShardedStreamDataset(cfg, rank=1, world_size=4)
    ds2 = ShardedStreamDataset(cfg, rank=2, world_size=4)
    ds3 = ShardedStreamDataset(cfg, rank=3, world_size=4)

    ids0 = set(b["sample_id"] for b in ds0)
    ids1 = set(b["sample_id"] for b in ds1)
    ids2 = set(b["sample_id"] for b in ds2)
    ids3 = set(b["sample_id"] for b in ds3)

    all_sets = [ids0, ids1, ids2, ids3]
    for i in range(4):
        for j in range(i + 1, 4):
            overlap = all_sets[i] & all_sets[j]
            if overlap:
                errors.append(
                    f"Rank {i} and rank {j} share {len(overlap)} samples"
                )

    # All samples covered
    union = ids0 | ids1 | ids2 | ids3
    if len(union) != cfg.num_samples:
        errors.append(
            f"Union covers {len(union)} samples, expected {cfg.num_samples}"
        )

    # Same seed+epoch gives same order
    ds_a = ShardedStreamDataset(cfg, rank=0, world_size=1)
    ds_a.set_epoch(5)
    order_a = [b["sample_id"] for b in ds_a]

    ds_b = ShardedStreamDataset(cfg, rank=0, world_size=1)
    ds_b.set_epoch(5)
    order_b = [b["sample_id"] for b in ds_b]

    if order_a != order_b:
        errors.append("Same seed+epoch produced different ordering")

    # set_epoch changes ordering
    ds_c = ShardedStreamDataset(cfg, rank=0, world_size=1)
    ds_c.set_epoch(0)
    order_c = [b["sample_id"] for b in ds_c]
    ds_c.set_epoch(1)
    order_d = [b["sample_id"] for b in ds_c]

    if order_c == order_d:
        errors.append("set_epoch did not change ordering")

    _print_gate_result("Gate 4", errors)
    return len(errors) == 0


def _print_gate_result(gate_name: str, errors: list) -> None:
    if errors:
        print(f"\n  RESULT: {gate_name} FAILED")
        for e in errors:
            print(f"    - {e}")
    else:
        print(f"\n  RESULT: {gate_name} PASSED")


def main():
    parser = argparse.ArgumentParser(description="Validate pipeline done-when gates")
    parser.add_argument("--gate", type=int, choices=[1, 2, 3, 4],
                        help="Run specific gate (1-4)")
    parser.add_argument("--all", action="store_true", default=True,
                        help="Run all gates (default)")
    args = parser.parse_args()

    gates = {
        1: validate_gate_1,
        2: validate_gate_2,
        3: validate_gate_3,
        4: validate_gate_4,
    }

    results = {}
    if args.gate:
        results[args.gate] = gates[args.gate]()
    else:
        for gate_num, gate_fn in gates.items():
            results[gate_num] = gate_fn()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    all_passed = True
    for gate_num, passed in sorted(results.items()):
        status = "PASSED" if passed else "FAILED"
        print(f"  Gate {gate_num}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n  All gates PASSED.")
    else:
        print("\n  Some gates FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
