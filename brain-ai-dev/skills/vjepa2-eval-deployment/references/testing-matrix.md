# Testing Matrix — Reference

## Overview

This matrix enumerates every test scenario required to validate the V-JEPA 2
assessment pipeline. Each row maps a scenario to its test class, assertion type,
and the done-when gate it satisfies.

## Done-When Gates Summary

| Gate | Description | Key Assertion |
|------|-------------|---------------|
| G1   | Frozen Probing | encoder grad is None; probe loss decreases |
| G2   | Focal Loss | matches expected value on synthetic data |
| G3   | Hub Loading | factory returns valid shapes |

---

## 1. AttentiveClassifier Tests

| # | Scenario | Expected | Assertion Type |
|---|----------|----------|----------------|
| 1.1 | Standard forward pass [B=4, N=196, D=128] -> [4, 10] | shape (4, 10) | assertEqual |
| 1.2 | Batch size 1 | shape (1, 10) | assertEqual |
| 1.3 | Batch size 32 | shape (32, 10) | assertEqual |
| 1.4 | num_queries=3 still produces [B, C] | shape (4, 10) | assertEqual |
| 1.5 | depth=4 (deeper pooler) | shape (4, 10) | assertEqual |
| 1.6 | num_heads=4 (multi-head attention) | shape (4, 10) | assertEqual |
| 1.7 | num_classes=1000 (ImageNet scale) | shape (4, 1000) | assertEqual |
| 1.8 | embed_dim=1408 (ViT-Giant scale) | shape (2, 174) | assertEqual |
| 1.9 | Gradient flows to classifier.head.weight | grad is not None | assertIsNotNone |
| 1.10 | Gradient flows to pooler.queries | grad is not None | assertIsNotNone |
| 1.11 | Encoder output is not modified | input unchanged | assert_close |
| 1.12 | Output dtype matches input dtype (float16) | dtype float16 | assertEqual |
| 1.13 | Deterministic output with same seed | out1 == out2 | assert_close |

---

## 2. FrozenBackboneAssessor Tests

| # | Scenario | Expected | Assertion Type | Gate |
|---|----------|----------|----------------|------|
| 2.1 | Encoder requires_grad stays False after training | all False | assertFalse | G1 |
| 2.2 | Encoder grad is None after backward | None | assertIsNone | G1 |
| 2.3 | Probe loss decreases over 20 steps | loss[-5:] < loss[:5] | assertLess | G1 |
| 2.4 | Multi-head: N=4 probes train simultaneously | 4 probe dicts | assertEqual | — |
| 2.5 | best_head() returns valid index in [0, N) | 0 <= idx < N | assertTrue | — |
| 2.6 | best_head() returns metrics dict with "accuracy" | key exists | assertIn | — |
| 2.7 | run_validation() returns per-head metrics | len == N | assertEqual | — |
| 2.8 | Probe parameters have non-None gradients | grad not None | assertIsNotNone | G1 |
| 2.9 | Second training run does not corrupt first | losses independent | assertTrue | — |
| 2.10 | val_only=True skips training | no optimizer step | Mock assert | — |

---

## 3. FocalLoss Tests

| # | Scenario | Expected | Assertion Type | Gate |
|---|----------|----------|----------------|------|
| 3.1 | FL on perfectly correct prediction -> ~0 | loss ≈ 0.0 | assert_close | G2 |
| 3.2 | FL on perfectly wrong prediction -> positive | loss > 0 | assertGreater | G2 |
| 3.3 | FL with gamma=0 matches cross-entropy * alpha | CE*alpha | assert_close | G2 |
| 3.4 | FL with alpha=1, gamma=2 known formula | manual calc | assert_close | G2 |
| 3.5 | FL is positive for any non-trivial input | loss > 0 | assertGreater | — |
| 3.6 | FL with gamma=5 < FL with gamma=0 for easy ex. | fl5 < fl0 | assertLess | — |
| 3.7 | FL batch size 1 runs without error | no exception | runs cleanly | — |
| 3.8 | FL batch size 64 matches batch size 1 average | mean equal | assert_close | — |
| 3.9 | FL handles all-same-class targets | no NaN/Inf | assertFalse(isnan) | — |
| 3.10 | FL handles num_classes=3806 (action space) | shape () scalar | assertEqual | — |

---

## 4. ClassMeanRecall Tests

| # | Scenario | Expected | Assertion Type |
|---|----------|----------|----------------|
| 4.1 | Perfect predictions -> recall = 1.0 | 1.0 | assert_close |
| 4.2 | All-wrong predictions -> recall = 0.0 | 0.0 | assert_close |
| 4.3 | 50% per-class accuracy -> recall = 0.5 | 0.5 | assert_close |
| 4.4 | Unbalanced classes: correct on all of rare class | recall in (0,1) | assertTrue |
| 4.5 | compute() result in [0.0, 1.0] | bounded | assertTrue |
| 4.6 | update() twice and compute() | cumulative | assertTrue |
| 4.7 | reset() clears counters | 0.0 after reset | assert_close |
| 4.8 | update_topk(k=5): ground truth in top-5 -> 1.0 | 1.0 | assert_close |
| 4.9 | update_topk(k=5): gt not in top-5 -> 0.0 | 0.0 | assert_close |
| 4.10 | Classes with no samples are excluded | no NaN | assertFalse |

---

## 5. ActionAnticipationClassifier Tests

| # | Scenario | Expected | Assertion Type |
|---|----------|----------|----------------|
| 5.1 | Forward returns tuple of 3 tensors | len==3 | assertEqual |
| 5.2 | verb logits shape [B, 97] | (4, 97) | assertEqual |
| 5.3 | noun logits shape [B, 300] | (4, 300) | assertEqual |
| 5.4 | action logits shape [B, 3806] | (4, 3806) | assertEqual |
| 5.5 | Gradients flow to all 3 heads | 3 non-None grads | assertIsNotNone |
| 5.6 | All 3 poolers have independent parameters | no shared weights | assertTrue |
| 5.7 | num_queries=3 still produces correct output shapes | as above | assertEqual |
| 5.8 | Works with predictor-shaped output [B, N_fut, D] | no error | runs cleanly |

---

## 6. Model Hub / Factory Tests

| # | Scenario | Expected | Assertion Type | Gate |
|---|----------|----------|----------------|------|
| 6.1 | vjepa2_vit_large(pretrained=False) returns tuple | len==2 | assertEqual | G3 |
| 6.2 | vjepa2_vit_giant(pretrained=False) returns tuple | len==2 | assertEqual | G3 |
| 6.3 | vjepa2_ac_vit_giant(pretrained=False) returns tuple | len==2 | assertEqual | G3 |
| 6.4 | encoder forward [2,3,16,224,224] -> [2,N,D] | 3D output | assertEqual | G3 |
| 6.5 | predictor forward produces 3D output | 3D output | assertEqual | G3 |
| 6.6 | encoder and predictor are nn.Module instances | isinstance | assertTrue | — |
| 6.7 | encoder output D matches expected embed_dim | D==1024 or 1408 | assertEqual | — |
| 6.8 | Factory without pretrained does not hit network | offline safe | no timeout | — |
| 6.9 | strict=False loading does not crash on RoPE | no KeyError | runs cleanly | — |

---

## 7. Preprocessor Tests

| # | Scenario | Expected | Assertion Type |
|---|----------|----------|----------------|
| 7.1 | vjepa2_preprocessor() returns Callable | callable | assertTrue |
| 7.2 | Preprocessor output shape [C, T, H, W] | (3, 16, 224, 224) | assertEqual |
| 7.3 | Preprocessor normalizes to ~zero mean | abs(mean) < 0.5 | assertLess |
| 7.4 | Preprocessor normalizes to ~unit std | 0.5 < std < 2.0 | assertTrue |
| 7.5 | Output dtype is float32 | torch.float32 | assertEqual |
| 7.6 | Pixel values NOT clipped to [0,1] after norm | may exceed 1 | assertTrue |
| 7.7 | crop_size=256 produces [3, 16, 256, 256] | (3, 16, 256, 256) | assertEqual |
| 7.8 | frames_per_clip=64 produces [3, 64, 224, 224] | (3, 64, 224, 224) | assertEqual |

---

## 8. AssessmentConfig Tests

| # | Scenario | Expected | Assertion Type |
|---|----------|----------|----------------|
| 8.1 | Default config has task="video_classification" | str | assertEqual |
| 8.2 | video_classification_ssv2() num_classes=174 | 174 | assertEqual |
| 8.3 | image_classification_imagenet() num_classes=1000 | 1000 | assertEqual |
| 8.4 | action_anticipation_epic() num_verbs=97 | 97 | assertEqual |
| 8.5 | action_anticipation_epic() num_nouns=300 | 300 | assertEqual |
| 8.6 | action_anticipation_epic() num_actions=3806 | 3806 | assertEqual |
| 8.7 | focal_alpha in (0, 1) | assertTrue | assertTrue |
| 8.8 | focal_gamma >= 0 | assertTrue | assertTrue |
| 8.9 | multihead_kwargs is empty list by default | len==0 | assertEqual |
| 8.10 | num_segments > 0 | assertTrue | assertTrue |

---

## 9. Multi-Head Selection Tests

| # | Scenario | Expected | Assertion Type |
|---|----------|----------|----------------|
| 9.1 | 4 heads trained; best_head idx consistent | in [0,4) | assertTrue |
| 9.2 | Head with higher LR may not always win | index varies | observational |
| 9.3 | run_validation returns exactly N metric dicts | len==4 | assertEqual |
| 9.4 | Metrics dict keys include standard metrics | "accuracy" in dict | assertIn |
| 9.5 | best_head after 0 steps returns a valid index | no crash | runs cleanly |

---

## 10. End-to-End Integration Tests (slow, GPU-conditional)

| # | Scenario | Expected | Assertion Type |
|---|----------|----------|----------------|
| 10.1 | Full frozen probing on synthetic SSv2 data | loss decreases | assertLess |
| 10.2 | Action anticipation 3-output forward + focal loss | loss > 0 | assertGreater |
| 10.3 | Multi-segment: 15 clips -> single prediction | shape [B, C] | assertEqual |
| 10.4 | AssessmentConfig preset round-trips through YAML | fields preserved | assertEqual |
| 10.5 | DDP wrapper on probe does not affect output shape | same shape | assertEqual |

---

## Coverage Target

- Unit tests (1–9): 100% of public API surface
- Integration tests (10): at least the 3 done-when gates
- Total test count target: 100+ individual test methods
