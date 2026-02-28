# Frozen Backbone Probing — Reference

## Overview

Frozen backbone probing is the standard protocol for assessing video representation quality.
The encoder weights are completely frozen; only a lightweight probe head is trained.
This isolates representation quality from downstream optimization.

## Core Pattern

```python
# 1. Load pretrained encoder
encoder = load_pretrained_vit()

# 2. Freeze all encoder parameters
for param in encoder.parameters():
    param.requires_grad = False

# 3. Put encoder in inference mode (avoids dropout, batchnorm updates)
#    Use .train(False) — equivalent to switching to non-training mode
encoder.train(False)

# 4. Create trainable probe head
probe = AttentiveClassifier(
    embed_dim=encoder.embed_dim,
    num_classes=num_classes,
    num_queries=1,
    depth=1,
    num_heads=1,
)

# 5. Optimizer covers only probe parameters
optimizer = torch.optim.SGD(
    probe.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
    momentum=0.9,
)

# 6. Training loop
for batch_frames, batch_labels in train_loader:
    with torch.no_grad():
        features = encoder(batch_frames)   # frozen, no grad
    logits = probe(features)               # trainable head
    loss = F.cross_entropy(logits, batch_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## AttentivePooler Architecture

The AttentivePooler replaces global average pooling with cross-attention, allowing the
model to attend to task-relevant spatial-temporal patches.

```
Input: encoder_output  [B, N, D]   # N = num_patches, D = embed_dim
       queries          [1, Q, D]   # Q = num_queries (learnable)

Cross-attention:
    Q: queries projected   -> [B, Q, D]
    K: encoder patches     -> [B, N, D]
    V: encoder patches     -> [B, N, D]
    attn = softmax(Q @ K^T / sqrt(D)) @ V  -> [B, Q, D]

Output: pooled [B, Q * D] after flatten, or [B, D] if Q=1
```

Implementation detail: queries are `nn.Parameter` of shape `[1, num_queries, embed_dim]`,
broadcast across the batch dimension.

```python
class AttentivePooler(nn.Module):
    def __init__(self, embed_dim: int, num_queries: int = 1,
                 num_heads: int = 1, depth: int = 1):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        B = x.shape[0]
        q = self.queries.expand(B, -1, -1)   # [B, Q, D]
        q = self.norm_q(q)
        kv = self.norm_kv(x)
        out, _ = self.cross_attn(q, kv, kv)  # [B, Q, D]
        return out  # caller flattens / projects


class AttentiveClassifier(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int,
                 num_queries: int = 1, depth: int = 1, num_heads: int = 1):
        super().__init__()
        self.pooler = AttentivePooler(embed_dim, num_queries, num_heads, depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        pooled = self.pooler(encoder_output)  # [B, Q, D]
        pooled = pooled.mean(dim=1)           # [B, D]  (mean over queries)
        pooled = self.norm(pooled)
        return self.head(pooled)              # [B, num_classes]
```

## Multi-Head Hyperparameter Search

Train N probe heads simultaneously — one per (lr, wd) combination — and select the best.

```python
multihead_kwargs = [
    {"lr": 1e-3, "wd": 1e-4},
    {"lr": 2e-3, "wd": 1e-4},
    {"lr": 5e-4, "wd": 1e-3},
    {"lr": 1e-2, "wd": 0.0},
]

# Create N independent probe heads
probes = [AttentiveClassifier(embed_dim, num_classes) for _ in multihead_kwargs]
optimizers = [
    torch.optim.SGD(p.parameters(), lr=kw["lr"], weight_decay=kw["wd"], momentum=0.9)
    for p, kw in zip(probes, multihead_kwargs)
]

for batch_frames, batch_labels in train_loader:
    with torch.no_grad():
        features = encoder(batch_frames)
    for probe, opt in zip(probes, optimizers):
        logits = probe(features)
        loss = F.cross_entropy(logits, batch_labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

# Validation: select best probe
best_acc = 0.0
best_idx = 0
for i, probe in enumerate(probes):
    acc = run_validation(probe, val_loader)
    if acc > best_acc:
        best_acc = acc
        best_idx = i
```

## Multi-Segment Assessment

At test time, each video is represented by multiple temporal segments and spatial views.
Final prediction = average of logits across all (segment, view) clips.

```python
num_segments = 5        # temporal segments per video
num_views_per_segment = 3   # spatial crops per segment
total_clips = num_segments * num_views_per_segment  # = 15

# During inference: frames shape [B * total_clips, C, T, H, W]
with torch.no_grad():
    features = encoder(frames_all_clips)
    logits = probe(features)  # [B * total_clips, num_classes]

# Reshape and average
logits = logits.view(B, total_clips, num_classes)
avg_logits = logits.mean(dim=1)   # [B, num_classes]
preds = avg_logits.argmax(dim=-1)
```

## Distributed Training Considerations

```python
# Wrap probe (not encoder) in DDP
probe = torch.nn.parallel.DistributedDataParallel(probe, device_ids=[local_rank])

# Encoder stays unwrapped — it runs in no_grad context anyway
encoder = encoder.to(device)  # no DDP wrapper needed
```

## Scheduler

Cosine annealing with linear warmup is standard for probe training:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
```

## Checkpoint Pattern

```python
ckpt = {
    "probe_state": probe.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "epoch": epoch,
    "best_acc": best_acc,
}
torch.save(ckpt, "probe_checkpoint.pt")
```

## Known Gotchas

1. **Do not call encoder.train()** during training — this re-enables dropout and updates
   running statistics in batchnorm layers. Keep encoder in `.train(False)` mode throughout.
2. **Gradient accumulation** must be applied only to the probe optimizer, not the encoder.
3. **Feature normalization**: encoder output is typically not L2-normalized; the AttentivePooler
   LayerNorm handles scale normalization internally.
4. **Memory**: with a frozen encoder in inference mode, only probe gradients are stored.
   This makes frozen probing ~10x more memory-efficient than full fine-tuning.
