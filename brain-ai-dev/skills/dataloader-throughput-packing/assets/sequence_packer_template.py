"""
SequencePacker — Packs variable-length sequences for pretraining and SFT.

Supports:
  - Pretraining block builder: rolling buffer -> fixed-length blocks
  - SFT boundary-aware packing: cu_seqlens + position_ids + labels
  - Bucketing: group by length, pack within bucket
  - Padding ratio computation

Usage:
    cfg = PackingConfig(mode="sft_boundary_aware", target_seq_len=2048)
    packer = SequencePacker(cfg)
    batch = packer.pack_sft(examples)
    ratio = packer.padding_ratio(batch)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Iterator, Optional, Tuple, Dict

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import numpy as np


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PackingConfig:
    """Configuration for the SequencePacker."""
    mode: str = "none"  # none | pretrain_blocks | sft_boundary_aware
    target_seq_len: int = 2048
    bucket_boundaries: Tuple[int, ...] = (256, 512, 1024, 2048)
    boundary_aware: bool = True
    pad_token_id: int = 0
    label_ignore_id: int = -100


@dataclass
class TokenizedExample:
    """A single tokenized example for SFT packing."""
    input_ids: List[int]
    labels: List[int]  # -100 for tokens that should not contribute to loss


@dataclass
class PackedBlock:
    """A fixed-length block for pretraining."""
    input_ids: List[int]
    doc_boundaries: List[int]  # positions where documents start within block
    padding_count: int = 0


@dataclass
class PackedBatch:
    """A packed batch for SFT with cu_seqlens."""
    input_ids: List[int]           # flattened token ids
    cu_seqlens: List[int]          # [0, len1, len1+len2, ...], int32
    position_ids: List[int]        # reset at each boundary
    labels: List[int]              # per-token labels
    max_seqlen: int = 0            # max individual sequence length
    num_examples: int = 0          # number of packed examples
    padding_count: int = 0         # number of padding tokens appended


# ---------------------------------------------------------------------------
# SequencePacker
# ---------------------------------------------------------------------------

class SequencePacker:
    """Packs variable-length sequences for pretraining or SFT."""

    def __init__(self, cfg: PackingConfig):
        self.cfg = cfg

    # ---- Pretraining block builder ----

    def pack_pretrain(
        self, token_streams: Iterator[List[int]]
    ) -> Iterator[PackedBlock]:
        """
        Concatenate token streams into fixed-length blocks.

        Yields PackedBlock of exactly target_seq_len tokens (except possibly
        the last block which may have padding).
        """
        target = self.cfg.target_seq_len
        buffer: List[int] = []
        doc_boundaries: List[int] = [0]  # track where each doc starts in buffer
        current_offset = 0

        for doc_tokens in token_streams:
            if not doc_tokens:
                continue
            buffer.extend(doc_tokens)
            current_offset += len(doc_tokens)
            doc_boundaries.append(current_offset)

            while len(buffer) >= target:
                block_tokens = buffer[:target]
                buffer = buffer[target:]

                # Adjust boundaries for this block
                block_bounds = [b for b in doc_boundaries if b < target]
                # Shift remaining boundaries
                doc_boundaries = [
                    b - target for b in doc_boundaries if b >= target
                ]
                if not doc_boundaries or doc_boundaries[0] != 0:
                    doc_boundaries.insert(0, 0)
                current_offset -= target

                yield PackedBlock(
                    input_ids=block_tokens,
                    doc_boundaries=block_bounds,
                    padding_count=0,
                )

        # Emit final block with padding if buffer is non-empty
        if buffer:
            pad_count = target - len(buffer)
            block_tokens = buffer + [self.cfg.pad_token_id] * pad_count
            block_bounds = [b for b in doc_boundaries if b < len(buffer)]
            yield PackedBlock(
                input_ids=block_tokens,
                doc_boundaries=block_bounds,
                padding_count=pad_count,
            )

    # ---- SFT boundary-aware packing ----

    def pack_sft(self, examples: List[TokenizedExample]) -> PackedBatch:
        """
        Pack variable-length SFT examples into a single flattened batch.

        Produces cu_seqlens for varlen attention and position_ids that reset
        at each example boundary.
        """
        if not examples:
            return PackedBatch(
                input_ids=[], cu_seqlens=[0], position_ids=[],
                labels=[], max_seqlen=0, num_examples=0, padding_count=0,
            )

        all_input_ids: List[int] = []
        all_labels: List[int] = []
        all_position_ids: List[int] = []
        cu_seqlens: List[int] = [0]
        max_seqlen = 0

        for ex in examples:
            length = len(ex.input_ids)
            all_input_ids.extend(ex.input_ids)
            all_labels.extend(ex.labels)
            all_position_ids.extend(range(length))
            cu_seqlens.append(cu_seqlens[-1] + length)
            max_seqlen = max(max_seqlen, length)

        return PackedBatch(
            input_ids=all_input_ids,
            cu_seqlens=cu_seqlens,
            position_ids=all_position_ids,
            labels=all_labels,
            max_seqlen=max_seqlen,
            num_examples=len(examples),
            padding_count=0,
        )

    # ---- Bucketing ----

    def assign_bucket(self, length: int) -> int:
        """Assign a sequence length to the appropriate bucket."""
        for boundary in self.cfg.bucket_boundaries:
            if length <= boundary:
                return boundary
        return self.cfg.bucket_boundaries[-1]

    def bucket_and_pack_sft(
        self, examples: List[TokenizedExample]
    ) -> Dict[int, PackedBatch]:
        """Group examples into buckets by length, then pack within each bucket."""
        from collections import defaultdict
        bucketed: Dict[int, List[TokenizedExample]] = defaultdict(list)
        for ex in examples:
            bucket = self.assign_bucket(len(ex.input_ids))
            bucketed[bucket].append(ex)

        result: Dict[int, PackedBatch] = {}
        for bucket_len, bucket_examples in bucketed.items():
            result[bucket_len] = self.pack_sft(bucket_examples)
        return result

    # ---- Padding ratio ----

    def padding_ratio(self, batch: PackedBatch) -> float:
        """
        Compute padding ratio for a packed batch.
        padding_ratio = padded_tokens / (padded_tokens + real_tokens)
        """
        real_tokens = len(batch.input_ids) - batch.padding_count
        total_tokens = len(batch.input_ids)
        if total_tokens == 0:
            return 0.0
        return batch.padding_count / total_tokens

    def padding_ratio_blocks(self, blocks: List[PackedBlock]) -> float:
        """Compute padding ratio across multiple pretraining blocks."""
        total_padding = sum(b.padding_count for b in blocks)
        total_tokens = sum(len(b.input_ids) for b in blocks)
        if total_tokens == 0:
            return 0.0
        return total_padding / total_tokens


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:
    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        status = "PASS" if condition else "FAIL"
        if not condition:
            failed += 1
            print(f"  [{status}] {name}: {detail}")
        else:
            passed += 1
            print(f"  [{status}] {name}")

    print("=" * 60)
    print("SequencePacker Self-Tests")
    print("=" * 60)

    cfg = PackingConfig(
        mode="pretrain_blocks",
        target_seq_len=100,
        bucket_boundaries=(50, 100, 200),
        pad_token_id=0,
    )
    packer = SequencePacker(cfg)

    # --- Pretrain block tests ---

    # Test 1: Pretrain blocks are exactly target_seq_len
    docs = [[1] * 50, [2] * 80, [3] * 120]
    blocks = list(packer.pack_pretrain(iter(docs)))
    for i, block in enumerate(blocks):
        check(f"T1: block {i} length == target_seq_len",
              len(block.input_ids) == cfg.target_seq_len,
              f"len={len(block.input_ids)}")

    # Test 2: Near-zero padding (only last block may have padding)
    non_last_padding = sum(b.padding_count for b in blocks[:-1])
    check("T2: non-last blocks have zero padding", non_last_padding == 0,
          f"non_last_padding={non_last_padding}")

    # Test 3: Total tokens preserved (including padding in last block)
    total_input = sum(len(d) for d in docs)
    total_output_real = sum(len(b.input_ids) - b.padding_count for b in blocks)
    check("T3: total tokens preserved", total_input == total_output_real,
          f"input={total_input}, output_real={total_output_real}")

    # Test 4: Doc boundaries tracked
    first_block = blocks[0]
    check("T4: doc_boundaries is a list",
          isinstance(first_block.doc_boundaries, list))
    check("T4b: doc_boundaries has entries",
          len(first_block.doc_boundaries) > 0,
          f"boundaries={first_block.doc_boundaries}")

    # --- SFT packing tests ---

    # Test 5: cu_seqlens starts at 0
    examples = [
        TokenizedExample(input_ids=[10, 20, 30], labels=[-100, 20, 30]),
        TokenizedExample(input_ids=[40, 50], labels=[40, 50]),
        TokenizedExample(input_ids=[60, 70, 80, 90], labels=[-100, -100, 80, 90]),
    ]
    batch = packer.pack_sft(examples)
    check("T5: cu_seqlens[0] == 0", batch.cu_seqlens[0] == 0,
          f"cu_seqlens[0]={batch.cu_seqlens[0]}")

    # Test 6: cu_seqlens is monotonically increasing
    mono = all(
        batch.cu_seqlens[i] < batch.cu_seqlens[i + 1]
        for i in range(len(batch.cu_seqlens) - 1)
    )
    check("T6: cu_seqlens monotonically increasing", mono,
          f"cu_seqlens={batch.cu_seqlens}")

    # Test 7: cu_seqlens[-1] == total tokens
    check("T7: cu_seqlens[-1] == total tokens",
          batch.cu_seqlens[-1] == len(batch.input_ids),
          f"cu_seqlens[-1]={batch.cu_seqlens[-1]}, total={len(batch.input_ids)}")

    # Test 8: position_ids reset at boundaries
    for ex_idx in range(len(examples)):
        start = batch.cu_seqlens[ex_idx]
        pos_at_start = batch.position_ids[start]
        check(f"T8: position_ids reset at boundary {ex_idx}",
              pos_at_start == 0,
              f"pos_ids[{start}]={pos_at_start}")

    # Test 9: position_ids are sequential within each example
    all_sequential = True
    for ex_idx in range(len(examples)):
        start = batch.cu_seqlens[ex_idx]
        end = batch.cu_seqlens[ex_idx + 1]
        expected_pos = list(range(end - start))
        actual_pos = batch.position_ids[start:end]
        if actual_pos != expected_pos:
            all_sequential = False
            break
    check("T9: position_ids sequential within examples", all_sequential)

    # Test 10: Labels preserved per example
    reconstructed_labels = []
    for ex in examples:
        reconstructed_labels.extend(ex.labels)
    check("T10: labels preserved",
          batch.labels == reconstructed_labels,
          f"len_batch={len(batch.labels)}, len_expected={len(reconstructed_labels)}")

    # Test 11: padding_ratio is 0 for perfectly packed batch
    ratio = packer.padding_ratio(batch)
    check("T11: padding_ratio == 0 for packed SFT",
          abs(ratio) < 1e-9,
          f"ratio={ratio}")

    # Test 12: padding_ratio correct for blocks with known padding
    padded_block = PackedBlock(
        input_ids=[1] * 80 + [0] * 20,
        doc_boundaries=[0],
        padding_count=20,
    )
    # Use PackedBatch wrapper for ratio computation
    padded_batch = PackedBatch(
        input_ids=padded_block.input_ids,
        cu_seqlens=[0, 80],
        position_ids=list(range(80)) + [0] * 20,
        labels=[1] * 80 + [-100] * 20,
        padding_count=20,
    )
    ratio2 = packer.padding_ratio(padded_batch)
    expected_ratio = 20.0 / 100.0  # 0.2
    check("T12: padding_ratio == 0.2 for 20/100 padding",
          abs(ratio2 - expected_ratio) < 1e-6,
          f"ratio={ratio2}, expected={expected_ratio}")

    # --- Bucketing tests ---

    # Test 13: Bucketing assigns to correct bucket
    check("T13a: 30 tokens -> bucket 50",
          packer.assign_bucket(30) == 50, f"got {packer.assign_bucket(30)}")
    check("T13b: 50 tokens -> bucket 50",
          packer.assign_bucket(50) == 50, f"got {packer.assign_bucket(50)}")
    check("T13c: 51 tokens -> bucket 100",
          packer.assign_bucket(51) == 100, f"got {packer.assign_bucket(51)}")
    check("T13d: 150 tokens -> bucket 200",
          packer.assign_bucket(150) == 200, f"got {packer.assign_bucket(150)}")
    check("T13e: 999 tokens -> bucket 200 (capped)",
          packer.assign_bucket(999) == 200, f"got {packer.assign_bucket(999)}")

    # Test 14: bucket_and_pack_sft groups correctly
    mixed_examples = [
        TokenizedExample(input_ids=list(range(30)), labels=list(range(30))),
        TokenizedExample(input_ids=list(range(80)), labels=list(range(80))),
        TokenizedExample(input_ids=list(range(40)), labels=list(range(40))),
    ]
    bucketed = packer.bucket_and_pack_sft(mixed_examples)
    check("T14: bucketing produces buckets", len(bucketed) > 0,
          f"num_buckets={len(bucketed)}")
    # 30 -> 50, 80 -> 100, 40 -> 50 => should have buckets 50 and 100
    check("T14b: bucket 50 has 2 examples",
          50 in bucketed and bucketed[50].num_examples == 2,
          f"buckets={list(bucketed.keys())}")
    check("T14c: bucket 100 has 1 example",
          100 in bucketed and bucketed[100].num_examples == 1,
          f"buckets={list(bucketed.keys())}")

    # Test 15: Empty examples produce empty batch
    empty_batch = packer.pack_sft([])
    check("T15: empty pack_sft produces empty batch",
          len(empty_batch.input_ids) == 0 and empty_batch.cu_seqlens == [0])

    # Test 16: max_seqlen is correct
    check("T16: max_seqlen correct",
          batch.max_seqlen == max(len(ex.input_ids) for ex in examples),
          f"max_seqlen={batch.max_seqlen}")

    # Test 17: padding_ratio_blocks across multiple blocks
    blocks_for_ratio = [
        PackedBlock(input_ids=[1] * 100, doc_boundaries=[0], padding_count=0),
        PackedBlock(input_ids=[1] * 90 + [0] * 10, doc_boundaries=[0], padding_count=10),
    ]
    ratio3 = packer.padding_ratio_blocks(blocks_for_ratio)
    expected_ratio3 = 10.0 / 200.0  # 0.05
    check("T17: padding_ratio_blocks correct",
          abs(ratio3 - expected_ratio3) < 1e-6,
          f"ratio={ratio3}, expected={expected_ratio3}")

    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_self_tests()
