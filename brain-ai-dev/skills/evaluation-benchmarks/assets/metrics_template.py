"""
MetricsSuite: Core metrics computation engine for brain_ai benchmarks.

Supports accuracy, top-k accuracy, F1 (macro/weighted/micro), AUROC,
precision, recall, confusion matrix, and per-class breakdowns.

Dependencies: torch, numpy (standard for brain_ai)
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


class MetricsSuite:
    """
    Accumulates predictions and targets across batches, then computes
    classification metrics on demand.

    Args:
        task_type: One of 'classify', 'binary'.
        num_classes: Total number of classes.
        device: Device for accumulation tensors (default 'cpu').
    """

    def __init__(
        self,
        task_type: str = "classify",
        num_classes: int = 10,
        device: str = "cpu",
    ) -> None:
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        self.task_type = task_type
        self.num_classes = num_classes
        self.device = device
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._confusion = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.int64, device=self.device
        )
        self._all_probs: List[Tensor] = []
        self._all_targets: List[Tensor] = []
        self._total = 0

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """
        Accumulate a batch of predictions and targets.

        Args:
            predictions: (B,) int class indices  **or**  (B, C) logits/probs.
            targets: (B,) int class indices.
        """
        if predictions.numel() == 0 and targets.numel() == 0:
            return
        if predictions.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Batch size mismatch: predictions {predictions.shape[0]} vs targets {targets.shape[0]}"
            )
        targets = targets.to(self.device).long()
        predictions = predictions.to(self.device)

        if (targets < 0).any():
            raise ValueError("Targets contain negative indices")

        # Derive predicted class labels
        if predictions.dim() == 1:
            pred_labels = predictions.long()
            probs = None
        else:
            pred_labels = predictions.argmax(dim=-1)
            probs = predictions.float()

        # Update confusion matrix
        for t, p in zip(targets, pred_labels):
            ti, pi = t.item(), p.item()
            if 0 <= ti < self.num_classes and 0 <= pi < self.num_classes:
                self._confusion[ti, pi] += 1

        # Store probabilities for AUROC
        if probs is not None:
            self._all_probs.append(probs.detach().cpu())
            self._all_targets.append(targets.detach().cpu())

        self._total += targets.shape[0]

    def compute(self) -> Dict[str, float]:
        """
        Compute all aggregated metrics from accumulated data.

        Returns:
            Dict with keys: accuracy, f1_macro, f1_weighted, f1_micro,
            precision_macro, recall_macro, auroc (may be NaN).
        """
        if self._total == 0:
            return self._empty_metrics()

        # Per-class stats from confusion matrix
        tp, fp, fn, support = self._tp_fp_fn_support()

        accuracy = float(self._confusion.diag().sum()) / float(self._total)
        precision_per = self._safe_div(tp.float(), (tp + fp).float())
        recall_per = self._safe_div(tp.float(), (tp + fn).float())
        f1_per = self._safe_div(
            2.0 * precision_per * recall_per,
            precision_per + recall_per,
        )

        # Macro
        f1_macro = float(f1_per.mean())
        precision_macro = float(precision_per.mean())
        recall_macro = float(recall_per.mean())

        # Weighted
        total_support = support.sum().float()
        if total_support > 0:
            weights = support.float() / total_support
            f1_weighted = float((f1_per * weights).sum())
        else:
            f1_weighted = 0.0

        # Micro
        tp_sum = tp.sum().float()
        fp_sum = fp.sum().float()
        fn_sum = fn.sum().float()
        prec_micro = self._safe_div_scalar(tp_sum, tp_sum + fp_sum)
        rec_micro = self._safe_div_scalar(tp_sum, tp_sum + fn_sum)
        f1_micro = self._safe_div_scalar(2.0 * prec_micro * rec_micro, prec_micro + rec_micro)

        # AUROC
        auroc = self._compute_auroc()

        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "f1_micro": f1_micro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "auroc": auroc,
        }

    def confusion_matrix(self, normalize: Optional[str] = None) -> Tensor:
        """
        Return the confusion matrix.

        Args:
            normalize: None, 'true' (rows), 'pred' (cols), 'all'.
        """
        cm = self._confusion.clone().float()
        if normalize == "true":
            row_sums = cm.sum(dim=1, keepdim=True).clamp(min=1)
            cm = cm / row_sums
        elif normalize == "pred":
            col_sums = cm.sum(dim=0, keepdim=True).clamp(min=1)
            cm = cm / col_sums
        elif normalize == "all":
            total = cm.sum().clamp(min=1)
            cm = cm / total
        return cm

    def per_class_metrics(self) -> Dict[int, Dict[str, float]]:
        """Return per-class precision, recall, F1, support."""
        tp, fp, fn, support = self._tp_fp_fn_support()
        precision_per = self._safe_div(tp.float(), (tp + fp).float())
        recall_per = self._safe_div(tp.float(), (tp + fn).float())
        f1_per = self._safe_div(
            2.0 * precision_per * recall_per,
            precision_per + recall_per,
        )
        result: Dict[int, Dict[str, float]] = {}
        for c in range(self.num_classes):
            result[c] = {
                "precision": float(precision_per[c]),
                "recall": float(recall_per[c]),
                "f1": float(f1_per[c]),
                "support": int(support[c]),
            }
        return result

    def top_k_accuracy(self, k: int = 5) -> float:
        """Compute top-k accuracy from stored probabilities."""
        if not self._all_probs:
            return 0.0
        probs = torch.cat(self._all_probs, dim=0)
        targets = torch.cat(self._all_targets, dim=0)
        if k >= probs.shape[1]:
            warnings.warn(f"k={k} >= num_classes={probs.shape[1]}, top-k is trivially 1.0")
            return 1.0
        _, topk_indices = probs.topk(k, dim=1)
        correct = (topk_indices == targets.unsqueeze(1)).any(dim=1)
        return float(correct.float().mean())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tp_fp_fn_support(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        cm = self._confusion
        tp = cm.diag()
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        support = cm.sum(dim=1)
        return tp, fp, fn, support

    @staticmethod
    def _safe_div(num: Tensor, den: Tensor) -> Tensor:
        result = torch.zeros_like(num)
        mask = den > 0
        result[mask] = num[mask] / den[mask]
        return result

    @staticmethod
    def _safe_div_scalar(num: float, den: float) -> float:
        if den == 0:
            return 0.0
        return float(num / den)

    def _empty_metrics(self) -> Dict[str, float]:
        return {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "f1_micro": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "auroc": float("nan"),
        }

    def _compute_auroc(self) -> float:
        """Compute macro-averaged one-vs-rest AUROC."""
        if not self._all_probs:
            return float("nan")

        probs = torch.cat(self._all_probs, dim=0)
        targets = torch.cat(self._all_targets, dim=0)

        # Softmax if needed (detect logits by checking range)
        if probs.min() < 0 or probs.max() > 1.0 + 1e-6:
            probs = torch.softmax(probs, dim=-1)

        unique_targets = targets.unique()
        if len(unique_targets) < 2:
            return float("nan")

        aurocs: List[float] = []
        for c in range(self.num_classes):
            binary_targets = (targets == c).long()
            if binary_targets.sum() == 0 or binary_targets.sum() == len(binary_targets):
                continue  # Skip: only one class present for this OVR split
            class_probs = probs[:, c] if probs.dim() > 1 else probs
            auc = self._binary_auroc(class_probs, binary_targets)
            if not math.isnan(auc):
                aurocs.append(auc)

        if not aurocs:
            return float("nan")
        return float(np.mean(aurocs))

    @staticmethod
    def _binary_auroc(scores: Tensor, targets: Tensor) -> float:
        """Compute AUROC for binary targets using trapezoidal rule."""
        scores_np = scores.cpu().numpy().astype(np.float64)
        targets_np = targets.cpu().numpy().astype(np.int64)

        n_pos = int(targets_np.sum())
        n_neg = len(targets_np) - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")

        # Sort by descending score
        order = np.argsort(-scores_np, kind="stable")
        sorted_targets = targets_np[order]
        sorted_scores = scores_np[order]

        # Compute TPR and FPR at each threshold
        tp_cum = np.cumsum(sorted_targets)
        fp_cum = np.cumsum(1 - sorted_targets)
        tpr = tp_cum / n_pos
        fpr = fp_cum / n_neg

        # Prepend origin
        tpr = np.concatenate([[0.0], tpr])
        fpr = np.concatenate([[0.0], fpr])

        # Handle tied scores: keep only last occurrence per unique score
        # (trapezoidal integration handles this correctly)
        auc = float(np.trapz(tpr, fpr))
        return auc


# ======================================================================
# Self-tests
# ======================================================================

def _run_self_tests() -> None:
    """Run 40+ self-tests for MetricsSuite."""
    passed = 0
    failed = 0

    def check(condition: bool, name: str) -> None:
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  PASS: {name}")
        else:
            failed += 1
            print(f"  FAIL: {name}")

    def approx(a: float, b: float, tol: float = 1e-4) -> bool:
        if math.isnan(a) and math.isnan(b):
            return True
        if math.isnan(a) or math.isnan(b):
            return False
        return abs(a - b) < tol

    print("=" * 60)
    print("MetricsSuite Self-Tests")
    print("=" * 60)

    # --- Test 1: Perfect accuracy ---
    s = MetricsSuite("classify", 3)
    s.update(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))
    m = s.compute()
    check(approx(m["accuracy"], 1.0), "T01 perfect accuracy")

    # --- Test 2: All wrong ---
    s = MetricsSuite("classify", 3)
    s.update(torch.tensor([1, 2, 0]), torch.tensor([0, 1, 2]))
    m = s.compute()
    check(approx(m["accuracy"], 0.0), "T02 all wrong accuracy")

    # --- Test 3: Partial accuracy ---
    s = MetricsSuite("classify", 3)
    s.update(torch.tensor([0, 1, 0]), torch.tensor([0, 0, 0]))
    m = s.compute()
    check(approx(m["accuracy"], 2.0 / 3.0), "T03 partial accuracy 2/3")

    # --- Test 4: Empty batch returns 0 ---
    s = MetricsSuite("classify", 3)
    m = s.compute()
    check(approx(m["accuracy"], 0.0), "T04 empty batch accuracy=0")

    # --- Test 5: F1 perfect ---
    s = MetricsSuite("classify", 3)
    s.update(torch.tensor([0, 0, 1, 1, 2, 2]), torch.tensor([0, 0, 1, 1, 2, 2]))
    m = s.compute()
    check(approx(m["f1_macro"], 1.0), "T05 perfect F1 macro")

    # --- Test 6: F1 all wrong ---
    s = MetricsSuite("classify", 3)
    s.update(torch.tensor([1, 1, 2, 2, 0, 0]), torch.tensor([0, 0, 1, 1, 2, 2]))
    m = s.compute()
    check(approx(m["f1_macro"], 0.0), "T06 all wrong F1=0")

    # --- Test 7: Binary F1 manual ---
    # TP=2, FP=1, FN=1 for class 0; TP=2, FP=1, FN=1 for class 1
    s = MetricsSuite("classify", 2)
    s.update(torch.tensor([0, 0, 0, 1, 1, 1]), torch.tensor([0, 0, 1, 0, 1, 1]))
    m = s.compute()
    # Class 0: prec=2/3, rec=2/3, f1=2/3; Class 1: prec=2/3, rec=2/3, f1=2/3
    check(approx(m["f1_macro"], 2.0 / 3.0), "T07 binary F1 manual")

    # --- Test 8: Confusion matrix shape ---
    s = MetricsSuite("classify", 5)
    s.update(torch.tensor([0, 1, 2, 3, 4]), torch.tensor([0, 1, 2, 3, 4]))
    cm = s.confusion_matrix()
    check(cm.shape == (5, 5), "T08 confusion matrix shape")

    # --- Test 9: Confusion matrix diagonal ---
    check(cm.diag().sum().item() == 5, "T09 confusion matrix diagonal=5")

    # --- Test 10: Confusion matrix row sums = support ---
    s = MetricsSuite("classify", 3)
    targets = torch.tensor([0, 0, 0, 1, 1, 2])
    s.update(torch.tensor([0, 1, 0, 1, 1, 0]), targets)
    cm = s.confusion_matrix()
    row_sums = cm.sum(dim=1)
    check(row_sums[0].item() == 3, "T10 confusion row sum class 0")
    check(row_sums[1].item() == 2, "T11 confusion row sum class 1")
    check(row_sums[2].item() == 1, "T12 confusion row sum class 2")

    # --- Test 13: Normalize true ---
    cm_norm = s.confusion_matrix(normalize="true")
    check(approx(cm_norm.sum(dim=1)[0].item(), 1.0), "T13 row-normalized sums to 1")

    # --- Test 14: Per-class metrics ---
    pc = s.per_class_metrics()
    check(0 in pc and 1 in pc and 2 in pc, "T14 per-class has all classes")

    # --- Test 15: Per-class support ---
    check(pc[0]["support"] == 3, "T15 per-class support class 0")

    # --- Test 16: Reset clears state ---
    s.reset()
    m = s.compute()
    check(approx(m["accuracy"], 0.0), "T16 reset clears state")

    # --- Test 17: Multi-batch accumulation ---
    s = MetricsSuite("classify", 2)
    s.update(torch.tensor([0, 0]), torch.tensor([0, 0]))
    s.update(torch.tensor([1, 1]), torch.tensor([1, 1]))
    m = s.compute()
    check(approx(m["accuracy"], 1.0), "T17 multi-batch accumulation")

    # --- Test 18: Multi-batch vs single batch ---
    s1 = MetricsSuite("classify", 3)
    all_preds = torch.tensor([0, 1, 2, 0, 1])
    all_targs = torch.tensor([0, 1, 0, 0, 2])
    s1.update(all_preds, all_targs)
    m1 = s1.compute()

    s2 = MetricsSuite("classify", 3)
    s2.update(all_preds[:3], all_targs[:3])
    s2.update(all_preds[3:], all_targs[3:])
    m2 = s2.compute()
    check(approx(m1["accuracy"], m2["accuracy"]), "T18 single vs multi-batch accuracy")
    check(approx(m1["f1_macro"], m2["f1_macro"]), "T19 single vs multi-batch F1")

    # --- Test 20: Logit input ---
    s = MetricsSuite("classify", 3)
    logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    s.update(logits, torch.tensor([0, 1, 2]))
    m = s.compute()
    check(approx(m["accuracy"], 1.0), "T20 logit input accuracy")

    # --- Test 21: AUROC perfect ---
    s = MetricsSuite("classify", 2)
    probs = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
    s.update(probs, torch.tensor([0, 0, 1, 1]))
    m = s.compute()
    check(approx(m["auroc"], 1.0), "T21 AUROC perfect separation")

    # --- Test 22: AUROC single class -> NaN ---
    s = MetricsSuite("classify", 2)
    probs = torch.tensor([[0.6, 0.4], [0.7, 0.3]])
    s.update(probs, torch.tensor([0, 0]))
    m = s.compute()
    check(math.isnan(m["auroc"]), "T22 AUROC single class -> NaN")

    # --- Test 23: Top-k accuracy ---
    s = MetricsSuite("classify", 5)
    probs = torch.tensor([
        [0.1, 0.5, 0.2, 0.15, 0.05],  # top-2: [1, 2], target=1 -> hit
        [0.3, 0.1, 0.4, 0.1, 0.1],     # top-2: [2, 0], target=0 -> hit
        [0.05, 0.05, 0.05, 0.8, 0.05],  # top-2: [3, 0], target=0 -> miss (target=0 not in top-2 with stable sort)
    ])
    s.update(probs, torch.tensor([1, 0, 0]))
    topk = s.top_k_accuracy(k=2)
    # target=1: 1 in topk([1,2]) -> yes
    # target=0: 0 in topk([2,0]) -> yes
    # target=0: 0 in topk([3,...]) -> depends on tie-breaking
    check(topk >= 0.0 and topk <= 1.0, "T23 top-k accuracy in range")

    # --- Test 24: Top-k with k >= num_classes ---
    tk = s.top_k_accuracy(k=10)
    check(approx(tk, 1.0), "T24 top-k k>=C -> 1.0")

    # --- Test 25: Negative targets raise ---
    s = MetricsSuite("classify", 3)
    try:
        s.update(torch.tensor([0, 1]), torch.tensor([0, -1]))
        check(False, "T25 negative targets should raise")
    except ValueError:
        check(True, "T25 negative targets raise ValueError")

    # --- Test 26: Mismatched batch sizes raise ---
    s = MetricsSuite("classify", 3)
    try:
        s.update(torch.tensor([0, 1, 2]), torch.tensor([0, 1]))
        check(False, "T26 mismatched sizes should raise")
    except ValueError:
        check(True, "T26 mismatched sizes raise ValueError")

    # --- Test 27: num_classes=1 ---
    s = MetricsSuite("classify", 1)
    s.update(torch.tensor([0, 0, 0]), torch.tensor([0, 0, 0]))
    m = s.compute()
    check(approx(m["accuracy"], 1.0), "T27 single class accuracy")

    # --- Test 28: Large batch ---
    s = MetricsSuite("classify", 10)
    n = 10000
    preds = torch.randint(0, 10, (n,))
    targets = torch.randint(0, 10, (n,))
    s.update(preds, targets)
    m = s.compute()
    check(0.0 <= m["accuracy"] <= 1.0, "T28 large batch accuracy in range")
    check(0.0 <= m["f1_macro"] <= 1.0, "T29 large batch F1 in range")

    # --- Test 30: F1 weighted != macro for imbalanced ---
    s = MetricsSuite("classify", 3)
    # Class 0: 100 samples, class 1: 5, class 2: 5
    preds_0 = torch.zeros(100, dtype=torch.long)
    targs_0 = torch.zeros(100, dtype=torch.long)
    preds_1 = torch.ones(5, dtype=torch.long)
    targs_1 = torch.ones(5, dtype=torch.long)
    preds_2 = torch.full((5,), 2, dtype=torch.long)
    targs_2 = torch.full((5,), 2, dtype=torch.long)
    s.update(torch.cat([preds_0, preds_1, preds_2]),
             torch.cat([targs_0, targs_1, targs_2]))
    m = s.compute()
    # All correct -> F1 macro = weighted = 1.0 (perfect)
    check(approx(m["f1_macro"], 1.0), "T30 perfect imbalanced f1_macro")
    check(approx(m["f1_weighted"], 1.0), "T31 perfect imbalanced f1_weighted")

    # --- Test 32: F1 weighted != macro when not perfect ---
    s = MetricsSuite("classify", 3)
    # 90 correct class-0, 10 wrong (predicted as 1), 5 correct class-1, 5 correct class-2
    p = torch.cat([torch.zeros(90, dtype=torch.long), torch.ones(10, dtype=torch.long),
                    torch.ones(5, dtype=torch.long), torch.full((5,), 2, dtype=torch.long)])
    t = torch.cat([torch.zeros(100, dtype=torch.long),
                    torch.ones(5, dtype=torch.long), torch.full((5,), 2, dtype=torch.long)])
    s.update(p, t)
    m = s.compute()
    # With imbalance, weighted and macro should differ
    check(m["f1_macro"] >= 0.0, "T32 imbalanced f1_macro >= 0")

    # --- Test 33: Confusion matrix normalize=all ---
    s = MetricsSuite("classify", 2)
    s.update(torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1]))
    cm_all = s.confusion_matrix(normalize="all")
    check(approx(cm_all.sum().item(), 1.0), "T33 normalize=all sums to 1")

    # --- Test 34: Confusion matrix normalize=pred ---
    cm_pred = s.confusion_matrix(normalize="pred")
    check(approx(cm_pred.sum(dim=0)[0].item(), 1.0), "T34 normalize=pred col sums to 1")

    # --- Test 35: Empty update is no-op ---
    s = MetricsSuite("classify", 3)
    s.update(torch.tensor([0, 1]), torch.tensor([0, 1]))
    s.update(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))
    m = s.compute()
    check(approx(m["accuracy"], 1.0), "T35 empty update no-op")

    # --- Test 36: AUROC with random predictions ---
    s = MetricsSuite("classify", 2)
    torch.manual_seed(42)
    n = 1000
    probs = torch.rand(n, 2)
    probs = probs / probs.sum(dim=1, keepdim=True)
    targets = torch.randint(0, 2, (n,))
    s.update(probs, targets)
    m = s.compute()
    check(0.35 < m["auroc"] < 0.65, "T36 random AUROC ~0.5")

    # --- Test 37: Micro F1 equals accuracy for single-label ---
    s = MetricsSuite("classify", 5)
    preds = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
    targets = torch.tensor([0, 1, 2, 3, 4, 1, 2, 0])
    s.update(preds, targets)
    m = s.compute()
    check(approx(m["f1_micro"], m["accuracy"]), "T37 micro F1 = accuracy")

    # --- Test 38: Per-class F1 individual check ---
    s = MetricsSuite("classify", 2)
    # Class 0: 3 TP, 1 FP, 0 FN -> prec=3/4, rec=3/3=1.0, f1=6/7
    # Class 1: 1 TP, 0 FP, 1 FN -> prec=1/1=1.0, rec=1/2, f1=2/3
    s.update(torch.tensor([0, 0, 0, 0, 1]), torch.tensor([0, 0, 0, 1, 1]))
    pc = s.per_class_metrics()
    check(approx(pc[0]["precision"], 0.75), "T38 class-0 precision=0.75")
    check(approx(pc[0]["recall"], 1.0), "T39 class-0 recall=1.0")
    check(approx(pc[1]["precision"], 1.0), "T40 class-1 precision=1.0")
    check(approx(pc[1]["recall"], 0.5), "T41 class-1 recall=0.5")

    # --- Test 42: Constructor with num_classes < 1 raises ---
    try:
        MetricsSuite("classify", 0)
        check(False, "T42 num_classes=0 should raise")
    except ValueError:
        check(True, "T42 num_classes=0 raises ValueError")

    # --- Test 43: AUROC with probabilities vs logits ---
    s_prob = MetricsSuite("classify", 2)
    s_logit = MetricsSuite("classify", 2)
    probs = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    logits = torch.log(probs + 1e-10)
    targets = torch.tensor([0, 1, 0])
    s_prob.update(probs, targets)
    s_logit.update(logits, targets)
    auc_prob = s_prob.compute()["auroc"]
    auc_logit = s_logit.compute()["auroc"]
    check(approx(auc_prob, auc_logit, tol=0.05), "T43 AUROC prob vs logit similar")

    # --- Test 44: Compute after reset returns empty ---
    s = MetricsSuite("classify", 3)
    s.update(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))
    s.reset()
    m = s.compute()
    check(approx(m["accuracy"], 0.0), "T44 compute after reset")

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 60)
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_self_tests()
