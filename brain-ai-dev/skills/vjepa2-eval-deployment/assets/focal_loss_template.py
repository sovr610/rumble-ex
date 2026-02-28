# Copyright (c) Meta Platforms, Inc. and affiliates.
# MIT License
#
# focal_loss_template.py
#
# FocalLoss, ClassMeanRecall, and ActionAnticipationClassifier.
# Used for EPIC-Kitchens 100 action anticipation probing.

from __future__ import annotations

import math
import unittest
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Import AttentivePooler from sibling asset
# ---------------------------------------------------------------------------
try:
    from attentive_classifier_template import AttentivePooler
except ImportError:
    # Inline fallback when running standalone
    class AttentivePooler(nn.Module):  # type: ignore[no-redef]
        def __init__(self, embed_dim, num_queries=1, num_heads=1, depth=1, dropout=0.0):
            super().__init__()
            self.embed_dim   = embed_dim
            self.num_queries = num_queries
            self.queries = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
            nn.init.trunc_normal_(self.queries, std=0.02)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=num_heads, batch_first=True
            )
            self.norm_q  = nn.LayerNorm(embed_dim)
            self.norm_kv = nn.LayerNorm(embed_dim)

        def forward(self, x: Tensor) -> Tensor:
            B  = x.shape[0]
            q  = self.queries.expand(B, -1, -1)
            q  = self.norm_q(q)
            kv = self.norm_kv(x)
            out, _ = self.cross_attn(q, kv, kv)
            return out  # [B, Q, D]


# ---------------------------------------------------------------------------
# FocalLoss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in action anticipation.

    Formula::

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where ``p_t`` is the probability assigned to the correct class.

    Args:
        alpha: Scalar weighting factor (0 < alpha <= 1). Default 0.25.
        gamma: Focusing parameter (>= 0). Higher values suppress easy examples.
               gamma=0 recovers alpha-weighted cross-entropy. Default 2.0.
        reduction: "mean" (default), "sum", or "none".
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none'")

        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute focal loss.

        Args:
            inputs:  [B, C] — unnormalized class logits.
            targets: [B]    — ground truth class indices (long tensor).

        Returns:
            Scalar loss (or [B] if reduction="none").
        """
        # Cross-entropy loss for each sample (no reduction) -> [B]
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # p_t: probability assigned to the correct class
        p_t = torch.exp(-ce_loss)  # [B]  (exp(-CE) = p_correct)

        # Focal weight: (1 - p_t)^gamma — suppresses easy examples
        focal_weight = (1.0 - p_t).pow(self.gamma)  # [B]

        # Weighted focal loss
        loss = self.alpha * focal_weight * ce_loss   # [B]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # "none"

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, gamma={self.gamma}, reduction={self.reduction!r}"


# ---------------------------------------------------------------------------
# ClassMeanRecall
# ---------------------------------------------------------------------------

class ClassMeanRecall:
    """
    Per-class recall metric with distributed (all_reduce) support.

    Class-Mean Recall = mean over all classes of (correct_k / total_k),
    where "correct_k" counts samples of class k correctly predicted.

    Supports both top-1 and top-K recall. For action anticipation, R@5 is
    standard (top-5 recall averaged per class).

    Usage::

        metric = ClassMeanRecall(num_classes=97)
        for logits, labels in val_loader:
            metric.update_topk(logits, labels, k=5)
        metric.all_reduce()   # only needed in distributed training
        print(metric.compute())
        metric.reset()
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes         = num_classes
        self.per_class_correct   = torch.zeros(num_classes)
        self.per_class_total     = torch.zeros(num_classes)

    def reset(self) -> None:
        """Clear all accumulated counts."""
        self.per_class_correct.zero_()
        self.per_class_total.zero_()

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """
        Update counters with top-1 predictions.

        Args:
            predictions: [B] predicted class indices.
            targets:     [B] ground truth class indices.
        """
        predictions = predictions.detach().cpu()
        targets     = targets.detach().cpu()

        for c in range(self.num_classes):
            mask = targets == c
            if mask.sum() > 0:
                self.per_class_correct[c] += (predictions[mask] == c).float().sum()
                self.per_class_total[c]   += mask.float().sum()

    def update_topk(self, logits: Tensor, targets: Tensor, k: int = 5) -> None:
        """
        Update counters with top-K predictions.

        Args:
            logits:  [B, C] unnormalized class scores.
            targets: [B] ground truth class indices.
            k:       Number of top predictions to check.
        """
        logits  = logits.detach().cpu()
        targets = targets.detach().cpu()

        topk_preds = logits.topk(min(k, logits.shape[1]), dim=-1).indices  # [B, k]

        for c in range(self.num_classes):
            mask = targets == c
            if mask.sum() > 0:
                in_topk = (topk_preds[mask] == c).any(dim=-1)  # [n_c]
                self.per_class_correct[c] += in_topk.float().sum()
                self.per_class_total[c]   += mask.float().sum()

    def compute(self) -> float:
        """
        Compute and return class-mean recall as a Python float.
        Classes with zero samples are excluded from the mean.
        """
        valid = self.per_class_total > 0
        if valid.sum() == 0:
            return 0.0
        recall_per_class = self.per_class_correct[valid] / self.per_class_total[valid]
        return recall_per_class.mean().item()

    def all_reduce(self) -> None:
        """
        Synchronize per-class counters across distributed ranks.
        No-op if distributed is not initialized.
        """
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(self.per_class_correct, op=dist.ReduceOp.SUM)
                dist.all_reduce(self.per_class_total,   op=dist.ReduceOp.SUM)
        except (ImportError, RuntimeError):
            pass

    def __repr__(self) -> str:
        return (
            f"ClassMeanRecall(num_classes={self.num_classes}, "
            f"current={self.compute():.4f})"
        )


# ---------------------------------------------------------------------------
# ActionAnticipationClassifier
# ---------------------------------------------------------------------------

class ActionAnticipationClassifier(nn.Module):
    """
    Multi-output classifier for EPIC-Kitchens 100 action anticipation.

    Uses three independent AttentivePooler heads to predict:
    - verb logits    [B, num_verbs]
    - noun logits    [B, num_nouns]
    - action logits  [B, num_actions]

    Typically receives the predictor output (future-frame latents), not
    encoder output directly.

    Args:
        embed_dim:   Dimensionality of predictor output tokens.
        num_verbs:   Number of verb classes (97 for EPIC-Kitchens 100).
        num_nouns:   Number of noun classes (300 for EPIC-Kitchens 100).
        num_actions: Number of action pair classes (3806 for EPIC-Kitchens 100).
        num_queries: Query tokens per pooler (default=3, one conceptually per task).
        num_heads:   Attention heads in each pooler.
        dropout:     Attention dropout.
    """

    def __init__(
        self,
        embed_dim: int,
        num_verbs: int,
        num_nouns: int,
        num_actions: int,
        num_queries: int = 3,
        num_heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim   = embed_dim
        self.num_verbs   = num_verbs
        self.num_nouns   = num_nouns
        self.num_actions = num_actions

        # Three fully independent poolers — each learns its own query
        self.verb_pooler   = AttentivePooler(embed_dim, num_queries=1, num_heads=num_heads, dropout=dropout)
        self.noun_pooler   = AttentivePooler(embed_dim, num_queries=1, num_heads=num_heads, dropout=dropout)
        self.action_pooler = AttentivePooler(embed_dim, num_queries=1, num_heads=num_heads, dropout=dropout)

        # Shared normalization before each head
        self.norm = nn.LayerNorm(embed_dim)

        # Classification heads
        self.verb_head   = nn.Linear(embed_dim, num_verbs)
        self.noun_head   = nn.Linear(embed_dim, num_nouns)
        self.action_head = nn.Linear(embed_dim, num_actions)

        # Initialize heads
        for head in (self.verb_head, self.noun_head, self.action_head):
            nn.init.trunc_normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)

    def forward(self, predictor_output: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            predictor_output: [B, N, D] — future latent tokens from predictor.

        Returns:
            Tuple of (verb_logits, noun_logits, action_logits):
                verb_logits   [B, num_verbs]
                noun_logits   [B, num_nouns]
                action_logits [B, num_actions]
        """
        x = self.norm(predictor_output)  # [B, N, D]

        verb_feat   = self.verb_pooler(x).squeeze(1)    # [B, D]
        noun_feat   = self.noun_pooler(x).squeeze(1)    # [B, D]
        action_feat = self.action_pooler(x).squeeze(1)  # [B, D]

        verb_logits   = self.verb_head(verb_feat)     # [B, num_verbs]
        noun_logits   = self.noun_head(noun_feat)     # [B, num_nouns]
        action_logits = self.action_head(action_feat) # [B, num_actions]

        return verb_logits, noun_logits, action_logits

    def compute_loss(
        self,
        verb_logits: Tensor,
        noun_logits: Tensor,
        action_logits: Tensor,
        verb_labels: Tensor,
        noun_labels: Tensor,
        action_labels: Tensor,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> Tensor:
        """
        Compute combined focal loss across all three tasks.

        Returns:
            Scalar loss = mean(verb_loss + noun_loss + action_loss).
        """
        loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        verb_loss   = loss_fn(verb_logits,   verb_labels)
        noun_loss   = loss_fn(noun_logits,   noun_labels)
        action_loss = loss_fn(action_logits, action_labels)
        return (verb_loss + noun_loss + action_loss) / 3.0


# ---------------------------------------------------------------------------
# Self-Tests
# ---------------------------------------------------------------------------

class _TestFocalLoss(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    def test_positive_loss_on_random_inputs(self):
        logits  = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss    = self.loss_fn(logits, targets)
        self.assertGreater(loss.item(), 0.0)

    def test_near_zero_loss_on_perfect_prediction(self):
        # Very high logit for correct class -> p_t ~= 1.0 -> FL ~= 0
        logits  = torch.zeros(2, 5)
        logits[0, 0] = 100.0
        logits[1, 2] = 100.0
        targets = torch.tensor([0, 2])
        loss    = self.loss_fn(logits, targets)
        self.assertLess(loss.item(), 1e-3)

    def test_gamma_zero_matches_alpha_ce(self):
        """gamma=0 should give alpha * cross_entropy."""
        loss_focal = FocalLoss(alpha=0.25, gamma=0.0)
        logits  = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        fl = loss_focal(logits, targets)
        ce = 0.25 * F.cross_entropy(logits, targets)
        torch.testing.assert_close(fl, ce, atol=1e-5, rtol=1e-5)

    def test_large_gamma_smaller_than_small_gamma_for_easy(self):
        """For easy examples (high p_t), larger gamma => smaller loss."""
        logits = torch.zeros(1, 2)
        logits[0, 0] = 10.0    # very confident for class 0
        targets = torch.tensor([0])
        fl_small = FocalLoss(alpha=1.0, gamma=0.5)(logits, targets)
        fl_large = FocalLoss(alpha=1.0, gamma=5.0)(logits, targets)
        self.assertLess(fl_large.item(), fl_small.item())

    def test_reduction_none_shape(self):
        loss_fn = FocalLoss(reduction="none")
        logits  = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss    = loss_fn(logits, targets)
        self.assertEqual(loss.shape, (4,))

    def test_no_nan_inf(self):
        logits  = torch.randn(16, 3806)
        targets = torch.randint(0, 3806, (16,))
        loss    = self.loss_fn(logits, targets)
        self.assertFalse(torch.isnan(loss).any())
        self.assertFalse(torch.isinf(loss).any())

    def test_invalid_alpha_raises(self):
        with self.assertRaises(ValueError):
            FocalLoss(alpha=0.0)

    def test_invalid_gamma_raises(self):
        with self.assertRaises(ValueError):
            FocalLoss(gamma=-1.0)

    def test_batch_size_one(self):
        logits  = torch.randn(1, 10)
        targets = torch.tensor([3])
        loss    = self.loss_fn(logits, targets)
        self.assertEqual(loss.shape, ())  # scalar


class _TestClassMeanRecall(unittest.TestCase):

    def setUp(self):
        self.metric = ClassMeanRecall(num_classes=5)

    def test_perfect_predictions(self):
        preds   = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        targets = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        self.metric.update(preds, targets)
        self.assertAlmostEqual(self.metric.compute(), 1.0, places=5)

    def test_all_wrong_predictions(self):
        preds   = torch.tensor([1, 0, 3, 2, 0])
        targets = torch.tensor([0, 1, 2, 3, 4])
        self.metric.update(preds, targets)
        self.assertAlmostEqual(self.metric.compute(), 0.0, places=5)

    def test_half_correct(self):
        # 2 samples per class, 1 correct each => recall = 0.5
        preds   = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4, 4, 0])
        targets = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        self.metric.update(preds, targets)
        self.assertAlmostEqual(self.metric.compute(), 0.5, places=5)

    def test_result_bounded(self):
        preds   = torch.randint(0, 5, (100,))
        targets = torch.randint(0, 5, (100,))
        self.metric.update(preds, targets)
        result = self.metric.compute()
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_reset_clears_state(self):
        preds   = torch.tensor([0, 1, 2])
        targets = torch.tensor([0, 1, 2])
        self.metric.update(preds, targets)
        self.metric.reset()
        self.assertAlmostEqual(self.metric.compute(), 0.0, places=5)

    def test_update_topk_perfect(self):
        num_classes = 5
        metric  = ClassMeanRecall(num_classes=num_classes)
        logits  = torch.eye(num_classes)   # [5, 5]: perfect logits
        targets = torch.arange(num_classes)
        metric.update_topk(logits, targets, k=1)
        self.assertAlmostEqual(metric.compute(), 1.0, places=5)

    def test_update_topk_worst(self):
        metric  = ClassMeanRecall(num_classes=5)
        # Ground truth always at position 5 (outside top-1 for 5 classes)
        logits  = torch.zeros(5, 10)
        # Put scores at classes 5-9, targets are 0-4
        logits[:, 5] = 10.0
        targets = torch.arange(5)
        metric.update_topk(logits, targets, k=4)
        # Class 5 is outside [0,5), so no match
        self.assertAlmostEqual(metric.compute(), 0.0, places=5)

    def test_classes_with_no_samples_excluded(self):
        metric = ClassMeanRecall(num_classes=10)
        # Only populate classes 0 and 1
        preds   = torch.tensor([0, 1])
        targets = torch.tensor([0, 1])
        metric.update(preds, targets)
        result = metric.compute()
        self.assertFalse(math.isnan(result))
        self.assertAlmostEqual(result, 1.0, places=5)


class _TestActionAnticipationClassifier(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.embed_dim   = 64
        self.batch_size  = 4
        self.num_patches = 16
        self.num_verbs   = 97
        self.num_nouns   = 300
        self.num_actions = 3806
        self.model = ActionAnticipationClassifier(
            embed_dim=self.embed_dim,
            num_verbs=self.num_verbs,
            num_nouns=self.num_nouns,
            num_actions=self.num_actions,
        )

    def _make_input(self):
        return torch.randn(self.batch_size, self.num_patches, self.embed_dim)

    def test_returns_three_tensors(self):
        out = self.model(self._make_input())
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 3)

    def test_verb_logits_shape(self):
        verb, noun, action = self.model(self._make_input())
        self.assertEqual(verb.shape, (self.batch_size, self.num_verbs))

    def test_noun_logits_shape(self):
        verb, noun, action = self.model(self._make_input())
        self.assertEqual(noun.shape, (self.batch_size, self.num_nouns))

    def test_action_logits_shape(self):
        verb, noun, action = self.model(self._make_input())
        self.assertEqual(action.shape, (self.batch_size, self.num_actions))

    def test_gradients_flow_to_all_heads(self):
        verb, noun, action = self.model(self._make_input())
        loss = verb.sum() + noun.sum() + action.sum()
        loss.backward()
        self.assertIsNotNone(self.model.verb_head.weight.grad)
        self.assertIsNotNone(self.model.noun_head.weight.grad)
        self.assertIsNotNone(self.model.action_head.weight.grad)

    def test_poolers_have_independent_queries(self):
        # Each pooler should have its own query parameter
        verb_q   = self.model.verb_pooler.queries
        noun_q   = self.model.noun_pooler.queries
        action_q = self.model.action_pooler.queries
        self.assertIsNot(verb_q, noun_q)
        self.assertIsNot(verb_q, action_q)

    def test_compute_loss_is_positive(self):
        x    = self._make_input()
        verb, noun, action = self.model(x)
        v_lbl = torch.randint(0, self.num_verbs,   (self.batch_size,))
        n_lbl = torch.randint(0, self.num_nouns,   (self.batch_size,))
        a_lbl = torch.randint(0, self.num_actions, (self.batch_size,))
        loss  = self.model.compute_loss(verb, noun, action, v_lbl, n_lbl, a_lbl)
        self.assertGreater(loss.item(), 0.0)

    def test_no_nan_in_outputs(self):
        verb, noun, action = self.model(self._make_input())
        for t, name in ((verb, "verb"), (noun, "noun"), (action, "action")):
            self.assertFalse(torch.isnan(t).any(), f"NaN in {name} logits")


if __name__ == "__main__":
    print("Running FocalLoss / ClassMeanRecall / ActionAnticipationClassifier self-tests...")
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromTestCase(_TestFocalLoss)
    suite.addTests(loader.loadTestsFromTestCase(_TestClassMeanRecall))
    suite.addTests(loader.loadTestsFromTestCase(_TestActionAnticipationClassifier))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\nAll self-tests passed.")
    else:
        raise SystemExit(1)
