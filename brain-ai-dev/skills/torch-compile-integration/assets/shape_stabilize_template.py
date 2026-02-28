"""
shape_stabilize_template.py
---------------------------
ShapeStabilizer: prevents torch.compile recompile thrashing from variable
sequence lengths by bucketing inputs to fixed shapes and providing
mark_dynamic utilities for pre-annotating dynamic dimensions.

Usage:
    from shape_stabilize_template import ShapeStabilizer

    stabilizer = ShapeStabilizer(buckets=[256, 512, 1024, 2048], pad_token_id=0)

    # In DataLoader collate_fn or training loop:
    padded_ids = stabilizer.bucket_batch(input_ids)  # shape: [B, bucket_size]

    # After bucketing, optionally mark dim 1 as dynamic within the bucket
    stabilizer.mark_dynamic_dims({"input_ids": padded_ids}, dim=1)

    # Unpad after model output
    original = stabilizer.unpad_batch(padded_ids, original_lengths)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ShapeStabilizer
# ---------------------------------------------------------------------------


class ShapeStabilizer:
    """
    Manages sequence-length bucketing to prevent torch.compile recompilation
    from variable-length inputs.

    Strategy
    --------
    Groups variable-length sequences into fixed "buckets" (size boundaries).
    Each batch is padded to the ceiling bucket for its maximum sequence length.
    This limits compiled specializations to len(buckets) shapes instead of one
    per unique length.

    Parameters
    ----------
    buckets:
        Sorted list of bucket boundary sizes. Must be strictly increasing
        positive integers. Sequences are padded to the smallest bucket
        >= their length.
    pad_token_id:
        Token ID used for padding positions. Should match the model's
        padding token. Padded positions must be masked in attention.
    """

    def __init__(
        self,
        buckets: List[int] = None,
        pad_token_id: int = 0,
    ) -> None:
        if buckets is None:
            buckets = [256, 512, 1024, 2048]

        if not buckets:
            raise ValueError("ShapeStabilizer: buckets list must be non-empty.")
        if sorted(buckets) != list(buckets):
            raise ValueError(
                f"ShapeStabilizer: buckets must be in ascending order. Got {buckets!r}."
            )
        if any(b <= 0 for b in buckets):
            raise ValueError("ShapeStabilizer: all bucket sizes must be positive integers.")
        if pad_token_id < 0:
            raise ValueError(
                f"ShapeStabilizer: pad_token_id must be >= 0. Got {pad_token_id}."
            )

        self.buckets: List[int] = list(buckets)
        self.pad_token_id: int = pad_token_id

    def bucket_size(self, seq_len: int) -> int:
        """
        Return the ceiling bucket for a given sequence length.

        The ceiling bucket is the smallest bucket value >= seq_len.

        Parameters
        ----------
        seq_len:
            Length of the sequence (number of tokens).

        Returns
        -------
        int
            The bucket size this sequence should be padded to.

        Raises
        ------
        ValueError
            If seq_len exceeds the largest bucket boundary.
        """
        if seq_len <= 0:
            raise ValueError(
                f"seq_len must be a positive integer. Got {seq_len}."
            )
        for b in self.buckets:
            if seq_len <= b:
                return b
        raise ValueError(
            f"seq_len={seq_len} exceeds the largest bucket {self.buckets[-1]}. "
            f"Add a larger bucket or truncate sequences to {self.buckets[-1]}."
        )

    def bucket_batch(self, input_ids: Tensor) -> Tensor:
        """
        Pad input_ids to the ceiling bucket for the batch's maximum length.

        All sequences in the batch are padded to the same bucket size.
        The bucket is determined by the longest sequence in the batch.

        Parameters
        ----------
        input_ids:
            2D tensor of shape [batch_size, seq_len] with dtype torch.long.
            Values should be token IDs. Existing padding is preserved.

        Returns
        -------
        Tensor
            Padded tensor of shape [batch_size, bucket_size] with dtype torch.long.
            New positions are filled with pad_token_id.

        Raises
        ------
        ValueError
            If input_ids is not 2D or the sequence length exceeds all buckets.
        """
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2D [batch_size, seq_len]. Got shape {input_ids.shape}."
            )

        batch_size, current_len = input_ids.shape

        if batch_size == 0:
            raise ValueError("input_ids batch_size must be > 0.")

        target_len = self.bucket_size(current_len)

        if target_len == current_len:
            return input_ids  # Already at bucket boundary

        # Pad on the right: (left_pad, right_pad) for last dim
        pad_amount = target_len - current_len
        padded = F.pad(input_ids, (0, pad_amount), mode="constant", value=self.pad_token_id)

        return padded

    def bucket_batch_variable(self, input_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Pad input_ids where sequences have variable lengths (right-padded internally).

        This variant tracks actual lengths and buckets based on maximum length
        in the batch. Useful when input_ids may already have padding.

        Parameters
        ----------
        input_ids:
            2D tensor [batch_size, max_seq_len]. Each row may already be
            padded with pad_token_id.

        Returns
        -------
        Tuple[Tensor, Tensor]:
            (padded_ids, original_lengths) where:
            - padded_ids has shape [batch_size, bucket_size]
            - original_lengths has shape [batch_size] with the non-padding length
              of each sequence
        """
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2D [batch_size, seq_len]. Got shape {input_ids.shape}."
            )

        # Compute original (non-padded) lengths
        original_lengths = (input_ids != self.pad_token_id).sum(dim=1)  # [B]
        max_original_len = original_lengths.max().item()

        # Determine bucket
        target_len = self.bucket_size(int(max_original_len))

        # Pad or truncate to bucket size
        current_len = input_ids.shape[1]
        if current_len < target_len:
            pad_amount = target_len - current_len
            padded = F.pad(
                input_ids, (0, pad_amount), mode="constant", value=self.pad_token_id
            )
        elif current_len > target_len:
            # Truncate to bucket (only if sequences fit within bucket)
            padded = input_ids[:, :target_len]
        else:
            padded = input_ids

        return padded, original_lengths

    def unpad_batch(
        self,
        padded: Tensor,
        original_lengths: Tensor,
    ) -> List[Tensor]:
        """
        Remove padding from a batched tensor using per-sequence original lengths.

        Parameters
        ----------
        padded:
            2D tensor [batch_size, padded_len].
        original_lengths:
            1D tensor [batch_size] of integers, the non-padded length of each sequence.

        Returns
        -------
        List[Tensor]
            List of tensors, each of shape [original_len_i] (1D, unpadded).
        """
        if padded.dim() != 2:
            raise ValueError(
                f"padded must be 2D [batch_size, padded_len]. Got shape {padded.shape}."
            )
        if original_lengths.dim() != 1:
            raise ValueError(
                f"original_lengths must be 1D [batch_size]. Got shape {original_lengths.shape}."
            )
        if padded.shape[0] != original_lengths.shape[0]:
            raise ValueError(
                f"Batch size mismatch: padded.shape[0]={padded.shape[0]} != "
                f"original_lengths.shape[0]={original_lengths.shape[0]}."
            )

        results = []
        for i in range(padded.shape[0]):
            length = int(original_lengths[i].item())
            results.append(padded[i, :length])

        return results

    def mark_dynamic_dims(
        self,
        batch: Dict[str, Tensor],
        dim: int = 1,
        max_len: Optional[int] = None,
    ) -> None:
        """
        Mark a dimension as dynamic on all tensors in the batch dict.

        This tells TorchDynamo to treat the specified dimension as having a
        variable (symbolic) size, preventing recompilation on shape changes
        in that dimension.

        IMPORTANT: Call this BEFORE invoking compiled code, on the input
        tensors for each step. Do NOT call inside the model's forward method.

        Parameters
        ----------
        batch:
            Dict of tensors (e.g., {"input_ids": ..., "attention_mask": ...}).
            Only Tensor values are processed; non-Tensor values are skipped.
        dim:
            Which dimension to mark as dynamic. Default 1 (sequence dimension
            in standard NLP: [batch, seq_len]).
        max_len:
            Optional upper bound for the dynamic dimension. Providing this
            allows the compiler to use range-based optimization.
            If None, uses the bucket size for the current batch.
        """
        try:
            import torch._dynamo
        except ImportError:
            logger.warning(
                "shape_stabilize: torch._dynamo not available — "
                "mark_dynamic_dims is a no-op."
            )
            return

        for key, value in batch.items():
            if not isinstance(value, Tensor):
                continue
            if value.dim() <= dim:
                logger.debug(
                    "shape_stabilize: skipping mark_dynamic on '%s' — "
                    "tensor has %d dims, dim=%d requested",
                    key, value.dim(), dim,
                )
                continue

            try:
                current_len = value.shape[dim]
                _max_len = max_len if max_len is not None else self.buckets[-1]

                torch._dynamo.mark_dynamic(
                    value,
                    dim=dim,
                    min=1,
                    max=_max_len,
                )
                logger.debug(
                    "shape_stabilize: marked '%s' dim=%d as dynamic (min=1, max=%d)",
                    key, dim, _max_len,
                )
            except Exception as e:
                logger.warning(
                    "shape_stabilize: failed to mark '%s' dim=%d as dynamic: %s",
                    key, dim, e,
                )

    def make_attention_mask(self, input_ids: Tensor) -> Tensor:
        """
        Generate an attention mask for a padded input_ids tensor.

        Returns a binary mask where 1 = real token, 0 = padding.

        Parameters
        ----------
        input_ids:
            2D tensor [batch_size, seq_len].

        Returns
        -------
        Tensor
            Binary attention mask [batch_size, seq_len] of dtype torch.long.
        """
        return (input_ids != self.pad_token_id).long()


# ---------------------------------------------------------------------------
# DataLoader-compatible collate function
# ---------------------------------------------------------------------------


def collate_fn_with_bucketing(
    batch: List[Dict[str, Any]],
    buckets: List[int] = None,
    pad_token_id: int = 0,
) -> Dict[str, Tensor]:
    """
    DataLoader collate_fn that pads sequences to bucket boundaries.

    Suitable for passing directly to DataLoader:
        from functools import partial
        collate = partial(collate_fn_with_bucketing, buckets=[256, 512, 1024, 2048])
        loader = DataLoader(dataset, collate_fn=collate)

    Parameters
    ----------
    batch:
        List of dicts from the Dataset's __getitem__. Each dict should contain
        "input_ids" as a list or 1D Tensor of token IDs. Other Tensor fields
        are padded to the same bucket size. Non-Tensor fields are stacked.
    buckets:
        Bucket boundaries. Defaults to [256, 512, 1024, 2048].
    pad_token_id:
        Token ID for padding positions.

    Returns
    -------
    Dict[str, Tensor]
        Batched dict with:
        - "input_ids": [batch_size, bucket_size]
        - "attention_mask": [batch_size, bucket_size]
        - Any other 1D Tensor fields from the batch, padded to bucket_size
        - "original_lengths": [batch_size] with actual sequence lengths
    """
    if buckets is None:
        buckets = [256, 512, 1024, 2048]

    stabilizer = ShapeStabilizer(buckets=buckets, pad_token_id=pad_token_id)

    # Extract input_ids (accept both list and Tensor)
    input_ids_list = []
    for item in batch:
        ids = item["input_ids"]
        if isinstance(ids, Tensor):
            ids = ids.long()
        else:
            ids = torch.tensor(ids, dtype=torch.long)
        input_ids_list.append(ids)

    # Compute original lengths
    original_lengths = torch.tensor(
        [len(ids) for ids in input_ids_list], dtype=torch.long
    )
    max_len = int(original_lengths.max().item())

    # Find bucket size
    bucket_sz = stabilizer.bucket_size(max_len)

    # Pad all input_ids to bucket_sz
    padded_ids = torch.full(
        (len(batch), bucket_sz), fill_value=pad_token_id, dtype=torch.long
    )
    for i, ids in enumerate(input_ids_list):
        length = min(len(ids), bucket_sz)
        padded_ids[i, :length] = ids[:length]

    attention_mask = (padded_ids != pad_token_id).long()

    result = {
        "input_ids": padded_ids,
        "attention_mask": attention_mask,
        "original_lengths": original_lengths,
    }

    # Handle other fields
    sample_keys = [k for k in batch[0].keys() if k != "input_ids"]
    for key in sample_keys:
        values = [item[key] for item in batch]
        # Convert list-of-ints / list-of-lists to Tensor for sequence fields
        if isinstance(values[0], (list, tuple)) and len(values[0]) > 0 and isinstance(values[0][0], (int, float)):
            values = [torch.tensor(v, dtype=torch.long) for v in values]
        if isinstance(values[0], Tensor) and values[0].dim() == 1:
            # 1D sequence field: pad like input_ids
            padded_field = torch.full(
                (len(batch), bucket_sz), fill_value=0, dtype=values[0].dtype
            )
            for i, v in enumerate(values):
                length = min(len(v), bucket_sz)
                padded_field[i, :length] = v[:length]
            result[key] = padded_field
        elif isinstance(values[0], Tensor):
            try:
                result[key] = torch.stack(values)
            except RuntimeError:
                result[key] = values  # type: ignore[assignment]
        elif isinstance(values[0], (int, float)):
            result[key] = torch.tensor(values)
        else:
            result[key] = values  # type: ignore[assignment]

    return result


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    failures: list = []

    def _check(name: str, condition: bool, msg: str = "") -> None:
        if condition:
            print(f"  PASS  {name}")
        else:
            print(f"  FAIL  {name}: {msg}")
            failures.append(name)

    print("=" * 60)
    print("ShapeStabilizer self-tests")
    print("=" * 60)

    BUCKETS = [256, 512, 1024, 2048]
    s = ShapeStabilizer(buckets=BUCKETS, pad_token_id=0)

    # Test 1-8: bucket_size correctness
    test_cases = [
        (1, 256),
        (100, 256),
        (256, 256),
        (257, 512),
        (512, 512),
        (513, 1024),
        (1024, 1024),
        (2048, 2048),
    ]
    for seq_len, expected_bucket in test_cases:
        result = s.bucket_size(seq_len)
        _check(
            f"bucket_size({seq_len})=={expected_bucket}",
            result == expected_bucket,
            f"Got {result}",
        )

    # Test 9: seq_len exceeds all buckets raises
    try:
        s.bucket_size(3000)
        _check("bucket_size_exceeds_raises", False, "Should have raised ValueError")
    except ValueError:
        _check("bucket_size_exceeds_raises", True)

    # Test 10: bucket_batch produces correct shape
    for seq_len in [100, 256, 400, 512, 700, 1024]:
        ids = torch.randint(1, 1000, (4, seq_len))
        padded = s.bucket_batch(ids)
        expected = s.bucket_size(seq_len)
        _check(
            f"bucket_batch_shape(seq_len={seq_len})",
            padded.shape == (4, expected),
            f"Got shape {padded.shape}, expected (4, {expected})",
        )

    # Test 11: padding values are correct (pad_token_id=0)
    ids = torch.ones(2, 100, dtype=torch.long)  # all 1s, no natural padding
    padded = s.bucket_batch(ids)
    _check(
        "padding_values_correct",
        padded[:, 100:].eq(0).all().item(),
        f"Expected 0 in padded positions, got {padded[:, 100:].unique().tolist()}",
    )

    # Test 12: original content is preserved after padding
    ids = torch.randint(1, 500, (3, 200))
    padded = s.bucket_batch(ids)
    _check(
        "original_content_preserved",
        padded[:, :200].equal(ids),
        "Original content not preserved after padding",
    )

    # Test 13: bucket_batch with exact boundary
    ids = torch.randint(1, 500, (4, 512))
    padded = s.bucket_batch(ids)
    _check(
        "bucket_batch_exact_boundary",
        padded.shape == (4, 512) and padded.equal(ids),
        f"Shape {padded.shape} or values changed at exact boundary",
    )

    # Test 14: unpad_batch recovers original sequences
    original_ids = [torch.randint(1, 500, (length,)) for length in [50, 120, 80]]
    padded_tensor = torch.full((3, 256), fill_value=0, dtype=torch.long)
    lengths = []
    for i, seq in enumerate(original_ids):
        padded_tensor[i, :len(seq)] = seq
        lengths.append(len(seq))
    original_lengths = torch.tensor(lengths)

    recovered = s.unpad_batch(padded_tensor, original_lengths)
    all_match = all(
        recovered[i].equal(original_ids[i]) for i in range(len(original_ids))
    )
    _check("unpad_recovers_original", all_match, "Unpadded sequences don't match originals")

    # Test 15: all bucket sizes are reachable
    for b in BUCKETS:
        result = s.bucket_size(b)
        _check(f"bucket_boundary_{b}_reachable", result == b, f"Got {result}")

    # Test 16: ShapeStabilizer rejects unsorted buckets
    try:
        _ = ShapeStabilizer(buckets=[1024, 256, 512])
        _check("unsorted_buckets_raises", False, "Should have raised ValueError")
    except ValueError:
        _check("unsorted_buckets_raises", True)

    # Test 17: make_attention_mask
    ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
    mask = s.make_attention_mask(ids)
    expected_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    _check(
        "make_attention_mask",
        mask.equal(expected_mask),
        f"Got {mask.tolist()}",
    )

    # Test 18: collate_fn_with_bucketing
    batch = [
        {"input_ids": list(range(100)), "labels": list(range(100))},
        {"input_ids": list(range(200)), "labels": list(range(200))},
        {"input_ids": list(range(50)), "labels": list(range(50))},
    ]
    result = collate_fn_with_bucketing(batch, buckets=BUCKETS, pad_token_id=0)
    _check(
        "collate_fn_shape",
        result["input_ids"].shape == (3, 256),
        f"Got shape {result['input_ids'].shape}",
    )
    _check(
        "collate_fn_has_attention_mask",
        "attention_mask" in result and result["attention_mask"].shape == (3, 256),
    )
    _check(
        "collate_fn_has_original_lengths",
        "original_lengths" in result
        and result["original_lengths"].tolist() == [100, 200, 50],
        f"Got {result.get('original_lengths', 'missing').tolist() if 'original_lengths' in result else 'missing'}",
    )

    print()
    if failures:
        print(f"FAILED: {len(failures)} tests: {failures}")
        sys.exit(1)
    else:
        n = 18 + len(test_cases) - 1  # test_cases contributes 8 tests, already counted
        # Count total: 8 bucket_size + 1 exceeds + 6 bucket_batch shapes + padding + content + exact + unpad + 4 boundaries + unsorted + mask + 3 collate = ~25+
        total = (
            len(test_cases)  # bucket_size tests
            + 1  # exceeds raises
            + 6  # bucket_batch shapes
            + 1  # padding values
            + 1  # content preserved
            + 1  # exact boundary
            + 1  # unpad
            + len(BUCKETS)  # bucket boundaries
            + 1  # unsorted raises
            + 1  # attention mask
            + 3  # collate fn
        )
        print(f"All {total} tests passed.")
