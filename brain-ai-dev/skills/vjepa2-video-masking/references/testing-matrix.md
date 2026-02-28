# Testing Matrix for V-JEPA 2 Video & Masking

## Overview

This document defines the complete test matrix for the video tokenization and
masking infrastructure.  Tests are organised by component and by the three
done-when gates specified in SKILL.md.

---

## Gate 1: Tokenization Shapes

### PatchEmbed (2-D)

| Input Shape       | patch_size | embed_dim | Expected Output      | Test ID    |
|-------------------|------------|-----------|----------------------|------------|
| [1, 3, 224, 224]  | 16         | 768       | [1, 196, 768]        | PE2D-001   |
| [4, 3, 224, 224]  | 16         | 768       | [4, 196, 768]        | PE2D-002   |
| [2, 3, 384, 384]  | 16         | 768       | [2, 576, 768]        | PE2D-003   |
| [1, 3, 224, 224]  | 32         | 1024      | [1, 49, 1024]        | PE2D-004   |
| [8, 1, 28, 28]    | 4          | 192       | [8, 49, 192]         | PE2D-005   |

### PatchEmbed3D

| Input Shape            | tubelet | patch | embed_dim | Expected Output    | Test ID    |
|------------------------|---------|-------|-----------|--------------------|------------|
| [1, 3, 16, 224, 224]   | 2       | 16    | 1024      | [1, 1568, 1024]    | PE3D-001   |
| [2, 3, 16, 256, 256]   | 2       | 16    | 1024      | [2, 2048, 1024]    | PE3D-002   |
| [4, 3, 8, 224, 224]    | 2       | 16    | 768       | [4, 784, 768]      | PE3D-003   |
| [1, 3, 16, 384, 384]   | 2       | 16    | 1024      | [1, 4608, 1024]    | PE3D-004   |
| [2, 3, 4, 224, 224]    | 2       | 16    | 1024      | [2, 392, 1024]     | PE3D-005   |
| [1, 3, 16, 224, 224]   | 4       | 16    | 1024      | [1, 784, 1024]     | PE3D-006   |

### Grid Size Properties

| Config                          | Expected grid_size   | Expected num_patches | Test ID  |
|---------------------------------|----------------------|----------------------|----------|
| frames=16, img=224, t=2, p=16   | (8, 14, 14)          | 1568                 | GS-001   |
| frames=16, img=256, t=2, p=16   | (8, 16, 16)          | 2048                 | GS-002   |
| frames=8,  img=224, t=2, p=16   | (4, 14, 14)          | 784                  | GS-003   |
| frames=16, img=224, t=4, p=16   | (4, 14, 14)          | 784                  | GS-004   |
| frames=16, img=224, t=2, p=32   | (8, 7, 7)            | 392                  | GS-005   |

### Divisibility Error Cases

| Config                         | Expected Behaviour      | Test ID  |
|--------------------------------|-------------------------|----------|
| frames=15, tubelet=2           | ValueError at __init__  | ERR-001  |
| img_size=225, patch_size=16    | ValueError at __init__  | ERR-002  |
| frames=16, tubelet=3           | ValueError at __init__  | ERR-003  |

---

## Gate 2: Mask Coverage

### Coverage Invariants (per sample)

| Property                                    | Formula                              | Test ID  |
|---------------------------------------------|--------------------------------------|----------|
| Full coverage (no gaps)                     | (enc OR pred).all()                  | COV-001  |
| No overlap                                  | NOT (enc AND pred).any()             | COV-002  |
| Exact complement                            | enc == NOT pred                      | COV-003  |
| Correct token count                         | enc.numel() == N = T*H*W             | COV-004  |
| enc count + pred count == N                 | enc.sum() + pred.sum() == N          | COV-005  |

### Shape Tests

| Config                          | batch_size | Expected masks_enc shape | Test ID  |
|---------------------------------|------------|--------------------------|----------|
| grid=(8,14,14), npred=8         | 1          | List[[1568]] bool        | SHP-001  |
| grid=(8,14,14), npred=8         | 4          | List[[1568]] x4          | SHP-002  |
| grid=(4,14,14), npred=4         | 2          | List[[784]] x2           | SHP-003  |
| grid=(8,16,16), npred=8         | 1          | List[[2048]] bool        | SHP-004  |

### Block Placement Tests

| Scenario                             | Assertion                         | Test ID  |
|--------------------------------------|-----------------------------------|----------|
| npred=1, large scale (0.7,0.7)       | pred.sum() / N >= 0.6             | BLK-001  |
| npred=8, small scale (0.15,0.15)     | pred.sum() / N > 0                | BLK-002  |
| temporal_scale=(1.0,1.0)             | pred covers all T frames          | BLK-003  |
| temporal_scale=(0.5,0.5)             | pred covers exactly T/2 frames    | BLK-004  |
| aspect_ratio=(1.0,1.0)               | block_h == block_w (square block) | BLK-005  |

### max_context_frames_ratio Tests

| ratio | T_grid | Expected max enc frame index | Test ID  |
|-------|--------|------------------------------|----------|
| 1.0   | 8      | 7 (all frames)               | MCF-001  |
| 0.5   | 8      | 3 (first 4 frames)           | MCF-002  |
| 0.25  | 8      | 1 (first 2 frames)           | MCF-003  |

### max_keep Tests

| N_total | max_keep | Assertion                   | Test ID  |
|---------|----------|-----------------------------|----------|
| 1568    | 500      | enc.sum() <= 500            | MK-001   |
| 1568    | None     | no cap applied              | MK-002   |
| 1568    | 2000     | enc unchanged (> N)         | MK-003   |
| 784     | 100      | enc.sum() <= 100            | MK-004   |

---

## Gate 3: Collator FPC Grouping

### Single-FPC Batch

| batch_size | FPC | Expected groups | Expected per-group batch size | Test ID  |
|------------|-----|-----------------|-------------------------------|----------|
| 4          | 16  | 1               | 4                             | FPC-001  |
| 8          | 8   | 1               | 8                             | FPC-002  |
| 16         | 16  | 1               | 16                            | FPC-003  |

### Multi-FPC Batch

| batch items FPC distribution | Expected groups | Group sizes | Test ID  |
|------------------------------|-----------------|-------------|----------|
| [16, 16, 8, 8]               | 2               | {16:2, 8:2} | FPC-004  |
| [16, 8, 4, 16, 8, 4]         | 3               | {16:2,8:2,4:2} | FPC-005 |
| [16, 16, 16, 16]             | 1               | {16:4}      | FPC-006  |
| [8, 8, 8, 16]                | 2               | {8:3, 16:1} | FPC-007  |

### Return Structure

| Test                                     | Assertion                            | Test ID  |
|------------------------------------------|--------------------------------------|----------|
| Output is a list                         | isinstance(result, list)             | COL-001  |
| Each element is a 3-tuple                | len(elem) == 3                       | COL-002  |
| masks_enc is list of tensors             | isinstance(m_enc, list)              | COL-003  |
| masks_pred is list of tensors            | isinstance(m_pred, list)             | COL-004  |
| Mask shape matches group token count     | m_enc[0].shape == (N_g,)            | COL-005  |
| Mask dtype is bool                       | m_enc[0].dtype == torch.bool         | COL-006  |

---

## Multi-Sequence Wrapper Tests

### MultiSequenceEncoder

| x_groups shapes           | masks per group | Expected output shapes    | Test ID  |
|---------------------------|-----------------|---------------------------|----------|
| [[2,1568,D], [2,784,D]]   | 50% visible     | [[2,784,D], [2,392,D]]    | MSE-001  |
| [[4,2048,D]]              | 25% visible     | [[4,512,D]]               | MSE-002  |
| [[1,1568,D], [3,1568,D]]  | same masks      | [[1,N_vis,D],[3,N_vis,D]] | MSE-003  |

### apply_masks

| x shape   | masks     | Expected output    | Test ID  |
|-----------|-----------|--------------------|----------|
| [2,10,D]  | 5 True    | [2,5,D]            | AM-001   |
| [1,100,D] | 75 True   | [1,75,D]           | AM-002   |
| [4,1568,D]| 392 True  | [4,392,D]          | AM-003   |
| [2,784,D] | all True  | [2,784,D]          | AM-004   |
| [2,784,D] | none True | [2,0,D]            | AM-005   |

### Correctness of Gathered Tokens

| Test                                           | Assertion                              | Test ID  |
|------------------------------------------------|----------------------------------------|----------|
| Gathered tokens match original at mask indices | x_vis[i] == x[i][m] for all i         | AM-010   |
| apply_masks_fast matches apply_masks           | torch.allclose(fast, naive)            | AM-011   |

---

## Seed Determinism Tests

| Scenario                                          | Assertion                               | Test ID  |
|---------------------------------------------------|-----------------------------------------|----------|
| Same seed -> identical masks_enc                  | masks_enc_a == masks_enc_b              | SD-001   |
| Different seeds -> different masks (prob.)        | masks_enc_a != masks_enc_b              | SD-002   |
| Counter increments across calls                   | seed_n+1 == seed_n + 1                  | SD-003   |
| Two workers share same counter                    | no duplicate seeds across workers       | SD-004   |
| Persistent workers: seeds continue from epoch N   | seeds not reset to 0 at epoch boundary  | SD-005   |

---

## Performance Baselines

| Scenario                                | Target                   | Test ID  |
|-----------------------------------------|--------------------------|----------|
| MaskGenerator: batch=32, grid=8x14x14  | < 10 ms per call         | PERF-001 |
| apply_masks: [32,1568,D] 75% masked    | < 5 ms per call          | PERF-002 |
| MaskCollator: batch=32 single FPC      | < 20 ms overhead vs base | PERF-003 |
| MaskCollator: batch=32 two FPC groups  | < 30 ms overhead vs base | PERF-004 |

---

## Edge Cases

| Scenario                                       | Expected Behaviour             | Test ID  |
|------------------------------------------------|--------------------------------|----------|
| npred=0 (no prediction blocks)                 | pred all False, enc all True   | EDGE-001 |
| npred large enough to cover all tokens         | pred may be all True           | EDGE-002 |
| batch_size=1                                   | single-element list returned   | EDGE-003 |
| grid=(1,1,1) (single token)                    | works without error            | EDGE-004 |
| max_keep=0                                     | enc all False                  | EDGE-005 |
| spatial_scale > 1.0                            | clamped to grid bounds         | EDGE-006 |
| aspect_ratio=(1e-3, 1e3)                       | block clamped to grid bounds   | EDGE-007 |
