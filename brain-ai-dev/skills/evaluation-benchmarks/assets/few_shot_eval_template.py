"""
FewShotEvaluator: N-way K-shot episode-based assessment for meta-learning.

Samples episodes, adapts the model on a support set, assesses on a query set,
and computes mean accuracy with 95% confidence interval.

Dependencies: torch, numpy (standard for brain_ai)
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class FewShotResult:
    """Result of a few-shot assessment run."""

    n_way: int
    k_shot: int
    n_query: int
    n_episodes: int
    mean_accuracy: float
    std_accuracy: float
    ci_95: float
    accuracy_low: float
    accuracy_high: float
    per_episode_accuracy: List[float] = field(default_factory=list)
    duration_seconds: float = 0.0
    collapsed_episodes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_way": self.n_way,
            "k_shot": self.k_shot,
            "n_query": self.n_query,
            "n_episodes": self.n_episodes,
            "mean_accuracy": self.mean_accuracy,
            "std_accuracy": self.std_accuracy,
            "ci_95": self.ci_95,
            "accuracy_low": self.accuracy_low,
            "accuracy_high": self.accuracy_high,
            "duration_seconds": self.duration_seconds,
            "collapsed_episodes": self.collapsed_episodes,
        }

    @property
    def is_collapsed(self) -> bool:
        """True if more than half of episodes collapsed to random."""
        return self.collapsed_episodes > self.n_episodes // 2


# ======================================================================
# Episode sampler
# ======================================================================

class EpisodeSampler:
    """
    Samples N-way K-shot episodes from a dataset organised as
    ``(features, labels)`` tensors.

    Args:
        features: (N_total, *feature_dims) tensor of all examples.
        labels:   (N_total,) int tensor of class labels.
        n_way:    Number of classes per episode.
        k_shot:   Support examples per class.
        n_query:  Query examples per class.
        seed:     Base random seed for reproducibility.
    """

    def __init__(
        self,
        features: Tensor,
        labels: Tensor,
        n_way: int = 5,
        k_shot: int = 1,
        n_query: int = 15,
        seed: int = 42,
    ) -> None:
        self.features = features
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.seed = seed

        # Build class-to-indices mapping
        self.class_indices: Dict[int, List[int]] = {}
        unique_labels = labels.unique().tolist()
        for c in unique_labels:
            self.class_indices[c] = (labels == c).nonzero(as_tuple=True)[0].tolist()

        self.available_classes = [
            c for c, idx in self.class_indices.items()
            if len(idx) >= k_shot + n_query
        ]
        if len(self.available_classes) < n_way:
            warnings.warn(
                f"Only {len(self.available_classes)} classes have enough samples "
                f"for {k_shot}-shot + {n_query}-query; need {n_way}."
            )

    def sample_episode(
        self, episode_idx: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Sample one episode.

        Returns:
            support_x  (n_way * k_shot, *feat_dims)
            support_y  (n_way * k_shot,)   -- relabelled to [0, n_way)
            query_x    (n_way * n_query, *feat_dims)
            query_y    (n_way * n_query,) -- relabelled to [0, n_way)
        """
        rng = np.random.RandomState(self.seed + episode_idx)
        classes = rng.choice(
            self.available_classes, size=self.n_way, replace=False
        ).tolist()

        support_indices: List[int] = []
        query_indices: List[int] = []
        support_labels: List[int] = []
        query_labels: List[int] = []

        for new_label, c in enumerate(classes):
            indices = self.class_indices[c]
            chosen = rng.choice(
                indices, size=self.k_shot + self.n_query, replace=False
            )
            s_idx = chosen[: self.k_shot].tolist()
            q_idx = chosen[self.k_shot :].tolist()
            support_indices.extend(s_idx)
            query_indices.extend(q_idx)
            support_labels.extend([new_label] * self.k_shot)
            query_labels.extend([new_label] * self.n_query)

        support_x = self.features[support_indices]
        query_x = self.features[query_indices]
        support_y = torch.tensor(support_labels, dtype=torch.long)
        query_y = torch.tensor(query_labels, dtype=torch.long)
        return support_x, support_y, query_x, query_y


# ======================================================================
# Default adaptation strategies
# ======================================================================

def prototype_adaptation(
    model: nn.Module,
    support_x: Tensor,
    support_y: Tensor,
    query_x: Tensor,
    n_way: int,
) -> Tensor:
    """
    Prototypical-network style adaptation: compute class prototypes from
    support set, return cosine-similarity logits for query set.
    """
    model.eval()
    with torch.no_grad():
        # Encode
        s_enc = _encode(model, support_x)  # (n_way*k_shot, D)
        q_enc = _encode(model, query_x)    # (n_way*n_query, D)

    # Compute prototypes
    prototypes = torch.zeros(n_way, s_enc.shape[-1], device=s_enc.device)
    for c in range(n_way):
        mask = support_y == c
        prototypes[c] = s_enc[mask].mean(dim=0)

    # Cosine similarity logits
    proto_norm = prototypes / prototypes.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    q_norm = q_enc / q_enc.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    logits = q_norm @ proto_norm.t()  # (Q, n_way)
    return logits


def _encode(model: nn.Module, x: Tensor) -> Tensor:
    """Pass *x* through *model*, returning a 2-D embedding."""
    out = model({"vision": x})
    if isinstance(out, dict):
        for k in ("embedding", "features", "logits", "output"):
            if k in out:
                out = out[k]
                break
        else:
            out = next(iter(out.values()))
    if out.dim() > 2:
        out = out.view(out.size(0), -1)
    return out


# ======================================================================
# FewShotEvaluator
# ======================================================================

class FewShotEvaluator:
    """
    N-way K-shot assessor.

    Args:
        model: torch.nn.Module used as feature extractor.
        n_way: Number of classes per episode.
        k_shot: Support examples per class.
        n_query: Query examples per class.
        adapt_fn: Callable (model, sx, sy, qx, n_way) -> logits.
    """

    def __init__(
        self,
        model: nn.Module,
        n_way: int = 5,
        k_shot: int = 1,
        n_query: int = 15,
        adapt_fn: Optional[Callable] = None,
    ) -> None:
        if n_way < 2:
            raise ValueError(f"n_way must be >= 2, got {n_way}")
        if k_shot < 1:
            raise ValueError(f"k_shot must be >= 1, got {k_shot}")
        if n_query < 1:
            raise ValueError(f"n_query must be >= 1, got {n_query}")
        self.model = model
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.adapt_fn = adapt_fn or prototype_adaptation

    def run_assessment(
        self,
        features: Tensor,
        labels: Tensor,
        n_episodes: int = 600,
        seed: int = 42,
    ) -> FewShotResult:
        """
        Run *n_episodes* few-shot episodes and return aggregated result.

        Args:
            features: (N, *dims) dataset features.
            labels:   (N,) class labels.
            n_episodes: Number of episodes to run.
            seed: Base random seed.
        """
        sampler = EpisodeSampler(
            features, labels,
            n_way=self.n_way, k_shot=self.k_shot, n_query=self.n_query,
            seed=seed,
        )
        if len(sampler.available_classes) < self.n_way:
            raise ValueError(
                f"Need >= {self.n_way} classes with at least "
                f"{self.k_shot + self.n_query} samples each; "
                f"got {len(sampler.available_classes)}."
            )

        episode_accs: List[float] = []
        collapsed = 0
        t0 = time.time()

        for ep in range(n_episodes):
            sx, sy, qx, qy = sampler.sample_episode(ep)
            logits = self.adapt_fn(self.model, sx, sy, qx, self.n_way)
            preds = logits.argmax(dim=-1)
            acc = float((preds == qy).float().mean())
            episode_accs.append(acc)

            # Detect collapse: all predictions the same class
            if preds.unique().numel() == 1:
                collapsed += 1

        duration = time.time() - t0
        arr = np.array(episode_accs)
        mean_acc = float(arr.mean())
        std_acc = float(arr.std())
        ci = 1.96 * std_acc / math.sqrt(max(n_episodes, 1))

        return FewShotResult(
            n_way=self.n_way,
            k_shot=self.k_shot,
            n_query=self.n_query,
            n_episodes=n_episodes,
            mean_accuracy=mean_acc,
            std_accuracy=std_acc,
            ci_95=ci,
            accuracy_low=mean_acc - ci,
            accuracy_high=mean_acc + ci,
            per_episode_accuracy=episode_accs,
            duration_seconds=duration,
            collapsed_episodes=collapsed,
        )

    # Alias for backward compatibility with SKILL.md contract
    evaluate = run_assessment


# ======================================================================
# Mock model for self-tests
# ======================================================================

class _MockEncoder(nn.Module):
    """Map input to a learned embedding."""

    def __init__(self, in_dim: int = 784, out_dim: int = 64) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, inputs: Dict[str, Tensor], **kw: Any) -> Tensor:
        x = list(inputs.values())[0]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.fc(x)


class _PerfectEncoder(nn.Module):
    """Returns one-hot vectors per true class (stored externally)."""

    def __init__(self, n_classes: int = 10, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_classes, embed_dim)
        # Make embeddings far apart -> easy prototypical classification
        with torch.no_grad():
            self.embed.weight.copy_(torch.eye(min(n_classes, embed_dim), embed_dim)[:n_classes])
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, inputs: Dict[str, Tensor], **kw: Any) -> Tensor:
        x = list(inputs.values())[0]
        # Treat first column as class hint
        idx = x[:, 0].long().clamp(0, self.embed.num_embeddings - 1)
        return self.embed(idx)


# ======================================================================
# Self-tests (25+)
# ======================================================================

def _run_self_tests() -> None:
    passed = 0
    failed = 0

    def check(cond: bool, name: str) -> None:
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  PASS: {name}")
        else:
            failed += 1
            print(f"  FAIL: {name}")

    def approx(a: float, b: float, tol: float = 0.05) -> bool:
        return abs(a - b) < tol

    print("=" * 60)
    print("FewShotEvaluator Self-Tests")
    print("=" * 60)

    # ---- Setup synthetic dataset: 20 classes, 50 samples each ----
    n_classes = 20
    samples_per_class = 50
    feat_dim = 32
    torch.manual_seed(42)
    np.random.seed(42)

    features_list = []
    labels_list = []
    for c in range(n_classes):
        centre = torch.randn(feat_dim) * 5
        xs = centre.unsqueeze(0) + torch.randn(samples_per_class, feat_dim) * 0.1
        features_list.append(xs)
        labels_list.append(torch.full((samples_per_class,), c, dtype=torch.long))
    all_features = torch.cat(features_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    # ---- FewShotResult dataclass ----
    fsr = FewShotResult(
        n_way=5, k_shot=1, n_query=15, n_episodes=100,
        mean_accuracy=0.65, std_accuracy=0.1, ci_95=0.02,
        accuracy_low=0.63, accuracy_high=0.67,
    )
    check(fsr.mean_accuracy == 0.65, "T01 FewShotResult field")
    d = fsr.to_dict()
    check("mean_accuracy" in d, "T02 to_dict has mean_accuracy")
    check("ci_95" in d, "T03 to_dict has ci_95")

    # ---- is_collapsed property ----
    fsr2 = FewShotResult(
        n_way=5, k_shot=1, n_query=15, n_episodes=100,
        mean_accuracy=0.2, std_accuracy=0.01, ci_95=0.001,
        accuracy_low=0.199, accuracy_high=0.201,
        collapsed_episodes=60,
    )
    check(fsr2.is_collapsed, "T04 is_collapsed when >50%")
    check(not fsr.is_collapsed, "T05 not collapsed by default")

    # ---- EpisodeSampler construction ----
    sampler = EpisodeSampler(
        all_features, all_labels,
        n_way=5, k_shot=1, n_query=15, seed=42,
    )
    check(len(sampler.available_classes) == n_classes, "T06 sampler has 20 classes")

    # ---- sample_episode shapes ----
    sx, sy, qx, qy = sampler.sample_episode(0)
    check(sx.shape == (5, feat_dim), "T07 support_x shape (5-way 1-shot)")
    check(sy.shape == (5,), "T08 support_y shape")
    check(qx.shape == (75, feat_dim), "T09 query_x shape (5*15)")
    check(qy.shape == (75,), "T10 query_y shape")

    # ---- Labels are in [0, n_way) ----
    check(sy.min().item() >= 0 and sy.max().item() < 5, "T11 support labels in [0,5)")
    check(qy.min().item() >= 0 and qy.max().item() < 5, "T12 query labels in [0,5)")

    # ---- Deterministic sampling ----
    sx2, sy2, _, _ = sampler.sample_episode(0)
    check(torch.equal(sx, sx2), "T13 same episode_idx same support")

    # ---- Different episodes differ ----
    sx3, sy3, _, _ = sampler.sample_episode(1)
    check(not torch.equal(sx, sx3), "T14 different episode_idx different support")

    # ---- FewShotEvaluator construction ----
    model = _MockEncoder(in_dim=feat_dim, out_dim=64)
    assessor = FewShotEvaluator(model, n_way=5, k_shot=1, n_query=15)
    check(assessor.n_way == 5, "T15 assessor n_way")
    check(assessor.k_shot == 1, "T16 assessor k_shot")

    # ---- basic run ----
    result = assessor.run_assessment(all_features, all_labels, n_episodes=20, seed=123)
    check(isinstance(result, FewShotResult), "T17 returns FewShotResult")
    check(result.n_episodes == 20, "T18 result n_episodes")
    check(0.0 <= result.mean_accuracy <= 1.0, "T19 mean_accuracy in [0,1]")
    check(result.ci_95 >= 0.0, "T20 ci_95 >= 0")
    check(result.duration_seconds >= 0.0, "T21 duration >= 0")
    check(len(result.per_episode_accuracy) == 20, "T22 per_episode list length")

    # ---- CI width decreases with more episodes ----
    res_few = assessor.run_assessment(all_features, all_labels, n_episodes=10, seed=1)
    res_many = assessor.run_assessment(all_features, all_labels, n_episodes=100, seed=1)
    check(res_many.ci_95 <= res_few.ci_95 + 0.1, "T23 more episodes -> tighter CI")

    # ---- 5-way 5-shot ----
    eval5 = FewShotEvaluator(model, n_way=5, k_shot=5, n_query=10)
    res5 = eval5.run_assessment(all_features, all_labels, n_episodes=20, seed=42)
    check(res5.k_shot == 5, "T24 5-shot result")

    # ---- 2-way assessment ----
    eval2 = FewShotEvaluator(model, n_way=2, k_shot=1, n_query=15)
    res2 = eval2.run_assessment(all_features, all_labels, n_episodes=20, seed=42)
    check(res2.n_way == 2, "T25 2-way result")

    # ---- Perfect encoder should get high accuracy ----
    perfect = _PerfectEncoder(n_classes=n_classes, embed_dim=feat_dim)
    # Encode class index into first column for the perfect encoder
    feat_with_hint = all_features.clone()
    for i in range(len(all_labels)):
        feat_with_hint[i, 0] = all_labels[i].float()

    eval_perf = FewShotEvaluator(perfect, n_way=5, k_shot=5, n_query=10)
    res_perf = eval_perf.run_assessment(feat_with_hint, all_labels, n_episodes=50, seed=42)
    check(res_perf.mean_accuracy > 0.85, "T26 perfect encoder high accuracy")

    # ---- Invalid n_way ----
    try:
        FewShotEvaluator(model, n_way=1)
        check(False, "T27 n_way=1 should raise")
    except ValueError:
        check(True, "T27 n_way=1 raises ValueError")

    # ---- Invalid k_shot ----
    try:
        FewShotEvaluator(model, n_way=5, k_shot=0)
        check(False, "T28 k_shot=0 should raise")
    except ValueError:
        check(True, "T28 k_shot=0 raises ValueError")

    # ---- Invalid n_query ----
    try:
        FewShotEvaluator(model, n_way=5, n_query=0)
        check(False, "T29 n_query=0 should raise")
    except ValueError:
        check(True, "T29 n_query=0 raises ValueError")

    # ---- accuracy_low <= mean <= accuracy_high ----
    check(result.accuracy_low <= result.mean_accuracy, "T30 accuracy_low <= mean")
    check(result.mean_accuracy <= result.accuracy_high, "T31 mean <= accuracy_high")

    # ---- collapsed_episodes ----
    check(result.collapsed_episodes >= 0, "T32 collapsed >= 0")

    # ---- Custom adapt_fn ----
    def random_adapt(model, sx, sy, qx, n_way):
        return torch.randn(qx.shape[0], n_way)

    eval_rand = FewShotEvaluator(model, n_way=5, k_shot=1, n_query=15, adapt_fn=random_adapt)
    res_rand = eval_rand.run_assessment(all_features, all_labels, n_episodes=50, seed=0)
    check(approx(res_rand.mean_accuracy, 0.2, tol=0.1), "T33 random adapt ~1/5 accuracy")

    # ---- to_dict round-trip fields ----
    d = result.to_dict()
    check(d["n_way"] == result.n_way, "T34 to_dict n_way")
    check(d["k_shot"] == result.k_shot, "T35 to_dict k_shot")
    check(d["collapsed_episodes"] == result.collapsed_episodes, "T36 to_dict collapsed")

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 60)
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_self_tests()
