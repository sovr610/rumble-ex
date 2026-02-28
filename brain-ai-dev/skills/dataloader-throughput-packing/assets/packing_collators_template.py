"""
Collator implementations for padded and packed batches.

Provides:
  - PaddingCollator: standard dynamic padding to max length in batch
  - PackingCollator: concatenate sequences, produce cu_seqlens + position_ids
  - PretrainBlockCollator: emit fixed-length token blocks
  - CollatorMetrics: track padding_ratio, effective_tokens, examples_per_batch

Usage:
    collator = PackingCollator(pad_token_id=0)
    batch = collator([example1, example2, example3])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# CollatorMetrics
# ---------------------------------------------------------------------------

@dataclass
class CollatorMetrics:
    """Metrics tracked by collators for each batch."""
    total_tokens: int = 0
    real_tokens: int = 0
    padding_tokens: int = 0
    examples_in_batch: int = 0

    @property
    def padding_ratio(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.padding_tokens / self.total_tokens

    @property
    def effective_tokens(self) -> int:
        return self.real_tokens


# ---------------------------------------------------------------------------
# PaddingCollator
# ---------------------------------------------------------------------------

class PaddingCollator:
    """
    Standard dynamic padding collator.

    Pads all sequences in a batch to the length of the longest sequence.
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        label_ignore_id: int = -100,
        max_length: Optional[int] = None,
    ):
        self.pad_token_id = pad_token_id
        self.label_ignore_id = label_ignore_id
        self.max_length = max_length
        self.last_metrics: Optional[CollatorMetrics] = None

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, Any]:
        """
        Pad a list of examples to the same length.

        Each example is a dict with "input_ids" and optionally "labels".
        """
        if not examples:
            self.last_metrics = CollatorMetrics()
            return {"input_ids": [], "labels": [], "attention_mask": []}

        lengths = [len(ex["input_ids"]) for ex in examples]
        max_len = min(max(lengths), self.max_length) if self.max_length else max(lengths)

        all_input_ids = []
        all_labels = []
        all_attention_mask = []
        total_padding = 0

        for ex in examples:
            ids = ex["input_ids"][:max_len]
            labs = ex.get("labels", ex["input_ids"])[:max_len]
            pad_len = max_len - len(ids)

            all_input_ids.append(ids + [self.pad_token_id] * pad_len)
            all_labels.append(labs + [self.label_ignore_id] * pad_len)
            all_attention_mask.append([1] * len(ids) + [0] * pad_len)
            total_padding += pad_len

        total_tokens = len(examples) * max_len
        real_tokens = total_tokens - total_padding
        self.last_metrics = CollatorMetrics(
            total_tokens=total_tokens,
            real_tokens=real_tokens,
            padding_tokens=total_padding,
            examples_in_batch=len(examples),
        )

        result = {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "attention_mask": all_attention_mask,
        }
        if HAS_TORCH:
            result = {k: torch.tensor(v, dtype=torch.long) for k, v in result.items()}
        return result


# ---------------------------------------------------------------------------
# PackingCollator
# ---------------------------------------------------------------------------

class PackingCollator:
    """
    Packing collator for SFT with boundary-aware attention.

    Concatenates sequences into a flat tensor, producing cu_seqlens and
    position_ids that reset at each sequence boundary.
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        label_ignore_id: int = -100,
        target_length: Optional[int] = None,
    ):
        self.pad_token_id = pad_token_id
        self.label_ignore_id = label_ignore_id
        self.target_length = target_length
        self.last_metrics: Optional[CollatorMetrics] = None

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, Any]:
        """
        Pack examples into a single flattened batch.

        Returns dict with input_ids, cu_seqlens, position_ids, labels, max_seqlen.
        """
        if not examples:
            self.last_metrics = CollatorMetrics()
            return {
                "input_ids": [] if not HAS_TORCH else torch.tensor([], dtype=torch.long),
                "cu_seqlens": [0] if not HAS_TORCH else torch.tensor([0], dtype=torch.int32),
                "position_ids": [] if not HAS_TORCH else torch.tensor([], dtype=torch.long),
                "labels": [] if not HAS_TORCH else torch.tensor([], dtype=torch.long),
                "max_seqlen": 0,
            }

        all_ids: List[int] = []
        all_labels: List[int] = []
        all_positions: List[int] = []
        cu_seqlens: List[int] = [0]
        max_seqlen = 0

        for ex in examples:
            ids = ex["input_ids"]
            labs = ex.get("labels", ids)
            length = len(ids)

            all_ids.extend(ids)
            all_labels.extend(labs)
            all_positions.extend(range(length))
            cu_seqlens.append(cu_seqlens[-1] + length)
            max_seqlen = max(max_seqlen, length)

        # Optional: pad to target_length
        padding_count = 0
        if self.target_length and len(all_ids) < self.target_length:
            padding_count = self.target_length - len(all_ids)
            all_ids.extend([self.pad_token_id] * padding_count)
            all_labels.extend([self.label_ignore_id] * padding_count)
            all_positions.extend([0] * padding_count)

        total_tokens = len(all_ids)
        real_tokens = total_tokens - padding_count
        self.last_metrics = CollatorMetrics(
            total_tokens=total_tokens,
            real_tokens=real_tokens,
            padding_tokens=padding_count,
            examples_in_batch=len(examples),
        )

        result: Dict[str, Any] = {
            "input_ids": all_ids,
            "cu_seqlens": cu_seqlens,
            "position_ids": all_positions,
            "labels": all_labels,
            "max_seqlen": max_seqlen,
        }

        if HAS_TORCH:
            result["input_ids"] = torch.tensor(all_ids, dtype=torch.long)
            result["cu_seqlens"] = torch.tensor(cu_seqlens, dtype=torch.int32)
            result["position_ids"] = torch.tensor(all_positions, dtype=torch.long)
            result["labels"] = torch.tensor(all_labels, dtype=torch.long)

        return result


# ---------------------------------------------------------------------------
# PretrainBlockCollator
# ---------------------------------------------------------------------------

class PretrainBlockCollator:
    """
    Collator that emits fixed-length token blocks for pretraining.

    Maintains a rolling buffer across calls. Emits blocks when the buffer
    reaches target_seq_len.
    """

    def __init__(
        self,
        target_seq_len: int = 2048,
        pad_token_id: int = 0,
    ):
        self.target_seq_len = target_seq_len
        self.pad_token_id = pad_token_id
        self._buffer: List[int] = []
        self.last_metrics: Optional[CollatorMetrics] = None

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, Any]:
        """
        Add examples to rolling buffer and emit fixed-length blocks.

        Returns dict with input_ids of shape [num_blocks, target_seq_len].
        """
        for ex in examples:
            self._buffer.extend(ex["input_ids"])

        blocks: List[List[int]] = []
        while len(self._buffer) >= self.target_seq_len:
            block = self._buffer[:self.target_seq_len]
            self._buffer = self._buffer[self.target_seq_len:]
            blocks.append(block)

        if not blocks:
            # Not enough tokens for a full block yet
            self.last_metrics = CollatorMetrics()
            if HAS_TORCH:
                return {
                    "input_ids": torch.zeros(0, self.target_seq_len, dtype=torch.long),
                    "labels": torch.zeros(0, self.target_seq_len, dtype=torch.long),
                }
            return {"input_ids": [], "labels": []}

        total_tokens = len(blocks) * self.target_seq_len
        self.last_metrics = CollatorMetrics(
            total_tokens=total_tokens,
            real_tokens=total_tokens,  # blocks have no padding
            padding_tokens=0,
            examples_in_batch=len(blocks),
        )

        result = {
            "input_ids": blocks,
            "labels": blocks,  # for pretraining, labels = input_ids shifted by model
        }
        if HAS_TORCH:
            result["input_ids"] = torch.tensor(blocks, dtype=torch.long)
            result["labels"] = torch.tensor(blocks, dtype=torch.long)
        return result

    def flush(self) -> Dict[str, Any]:
        """Flush remaining buffer as a padded block."""
        if not self._buffer:
            return {"input_ids": [], "labels": []}
        pad_count = self.target_seq_len - len(self._buffer)
        block = self._buffer + [self.pad_token_id] * pad_count
        self._buffer = []
        self.last_metrics = CollatorMetrics(
            total_tokens=self.target_seq_len,
            real_tokens=self.target_seq_len - pad_count,
            padding_tokens=pad_count,
            examples_in_batch=1,
        )
        result = {"input_ids": [block], "labels": [block]}
        if HAS_TORCH:
            result["input_ids"] = torch.tensor([block], dtype=torch.long)
            result["labels"] = torch.tensor([block], dtype=torch.long)
        return result


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
    print("PackingCollators Self-Tests")
    print("=" * 60)

    # ---- PaddingCollator tests ----

    pad_collator = PaddingCollator(pad_token_id=0, label_ignore_id=-100)

    # Test 1: Padding pads to max length
    examples = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
        {"input_ids": [4, 5, 6, 7, 8], "labels": [4, 5, 6, 7, 8]},
        {"input_ids": [9, 10], "labels": [9, 10]},
    ]
    batch = pad_collator(examples)
    if HAS_TORCH:
        ids = batch["input_ids"]
        check("T1: padding pads to max length",
              ids.shape[1] == 5, f"shape={ids.shape}")
    else:
        max_len = max(len(row) for row in batch["input_ids"])
        check("T1: padding pads to max length", max_len == 5, f"max_len={max_len}")

    # Test 2: Padding tokens are pad_token_id
    if HAS_TORCH:
        row0 = batch["input_ids"][0].tolist()
    else:
        row0 = batch["input_ids"][0]
    check("T2: pad tokens correct",
          row0 == [1, 2, 3, 0, 0], f"row0={row0}")

    # Test 3: Attention mask correct
    if HAS_TORCH:
        mask0 = batch["attention_mask"][0].tolist()
    else:
        mask0 = batch["attention_mask"][0]
    check("T3: attention mask correct",
          mask0 == [1, 1, 1, 0, 0], f"mask={mask0}")

    # Test 4: PaddingCollator metrics
    m = pad_collator.last_metrics
    check("T4: padding metrics total tokens",
          m.total_tokens == 15, f"total={m.total_tokens}")
    expected_padding = (5 - 3) + (5 - 5) + (5 - 2)  # 2 + 0 + 3 = 5
    check("T4b: padding metrics padding count",
          m.padding_tokens == expected_padding,
          f"padding={m.padding_tokens}, expected={expected_padding}")
    expected_ratio = expected_padding / 15.0
    check("T4c: padding ratio correct",
          abs(m.padding_ratio - expected_ratio) < 1e-6,
          f"ratio={m.padding_ratio}, expected={expected_ratio}")

    # ---- PackingCollator tests ----

    pack_collator = PackingCollator(pad_token_id=0)

    # Test 5: Packing produces valid cu_seqlens
    pack_examples = [
        {"input_ids": [10, 20, 30], "labels": [-100, 20, 30]},
        {"input_ids": [40, 50], "labels": [40, 50]},
        {"input_ids": [60, 70, 80, 90], "labels": [-100, -100, 80, 90]},
    ]
    pbatch = pack_collator(pack_examples)

    if HAS_TORCH:
        cu = pbatch["cu_seqlens"].tolist()
    else:
        cu = pbatch["cu_seqlens"]
    check("T5: cu_seqlens starts at 0", cu[0] == 0, f"cu={cu}")
    check("T5b: cu_seqlens ends at total",
          cu[-1] == 9, f"cu[-1]={cu[-1]}")  # 3+2+4=9

    # Test 6: cu_seqlens monotonically increasing
    mono = all(cu[i] < cu[i + 1] for i in range(len(cu) - 1))
    check("T6: cu_seqlens monotonically increasing", mono, f"cu={cu}")

    # Test 7: position_ids reset at boundaries
    if HAS_TORCH:
        pos = pbatch["position_ids"].tolist()
    else:
        pos = pbatch["position_ids"]
    for i in range(len(cu) - 1):
        check(f"T7: position reset at boundary {i}",
              pos[cu[i]] == 0,
              f"pos[{cu[i]}]={pos[cu[i]]}")

    # Test 8: position_ids sequential within each example
    all_seq = True
    for i in range(len(cu) - 1):
        start, end = cu[i], cu[i + 1]
        expected = list(range(end - start))
        actual = pos[start:end]
        if actual != expected:
            all_seq = False
            break
    check("T8: positions sequential within examples", all_seq)

    # Test 9: Packing metrics — zero padding
    pm = pack_collator.last_metrics
    check("T9: packing has zero padding",
          pm.padding_tokens == 0, f"padding={pm.padding_tokens}")
    check("T9b: packing ratio is 0",
          abs(pm.padding_ratio) < 1e-9, f"ratio={pm.padding_ratio}")

    # Test 10: PackingCollator with target_length adds padding
    pack_padded = PackingCollator(pad_token_id=0, target_length=20)
    pbatch2 = pack_padded(pack_examples)
    if HAS_TORCH:
        total_len = pbatch2["input_ids"].shape[0]
    else:
        total_len = len(pbatch2["input_ids"])
    check("T10: target_length pads to 20",
          total_len == 20, f"total_len={total_len}")
    pm2 = pack_padded.last_metrics
    check("T10b: padding count = 20 - 9 = 11",
          pm2.padding_tokens == 11, f"padding={pm2.padding_tokens}")

    # ---- PretrainBlockCollator tests ----

    block_collator = PretrainBlockCollator(target_seq_len=10, pad_token_id=0)

    # Test 11: Block collator emits fixed-length blocks
    block_examples = [
        {"input_ids": list(range(1, 26))},  # 25 tokens -> 2 full blocks, 5 in buffer
    ]
    bbatch = block_collator(block_examples)
    if HAS_TORCH:
        bids = bbatch["input_ids"]
        check("T11: block collator shape",
              bids.shape == (2, 10), f"shape={bids.shape}")
    else:
        check("T11: block collator emits 2 blocks",
              len(bbatch["input_ids"]) == 2,
              f"num_blocks={len(bbatch['input_ids'])}")

    # Test 12: Block collator has zero padding in full blocks
    bm = block_collator.last_metrics
    check("T12: block collator zero padding",
          bm.padding_tokens == 0, f"padding={bm.padding_tokens}")

    # Test 13: Block collator flush emits padded block
    flushed = block_collator.flush()
    if HAS_TORCH:
        fids = flushed["input_ids"]
        check("T13: flush emits one block",
              fids.shape == (1, 10), f"shape={fids.shape}")
        # Should be 5 real tokens + 5 padding
        check("T13b: flush has correct padding",
              fids[0, -1].item() == 0 and fids[0, 4].item() != 0)
    else:
        check("T13: flush emits one block",
              len(flushed["input_ids"]) == 1)

    # Test 14: Empty examples produce empty output
    empty_batch = pad_collator([])
    if HAS_TORCH:
        check("T14: empty padding collator",
              len(empty_batch["input_ids"]) == 0)
    else:
        check("T14: empty padding collator",
              empty_batch["input_ids"] == [])

    # Test 15: Labels preserved in packing collator
    if HAS_TORCH:
        labs = pbatch["labels"].tolist()
    else:
        labs = pbatch["labels"]
    expected_labs = [-100, 20, 30, 40, 50, -100, -100, 80, 90]
    check("T15: labels preserved in packing",
          labs == expected_labs, f"labels={labs}")

    # Test 16: max_seqlen correct
    check("T16: max_seqlen correct",
          pbatch["max_seqlen"] == 4, f"max_seqlen={pbatch['max_seqlen']}")

    # Test 17: CollatorMetrics effective_tokens property
    m_test = CollatorMetrics(total_tokens=100, real_tokens=80, padding_tokens=20)
    check("T17: effective_tokens correct",
          m_test.effective_tokens == 80)
    check("T17b: padding_ratio correct",
          abs(m_test.padding_ratio - 0.2) < 1e-9)

    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_self_tests()
