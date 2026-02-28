# Episodic Task Sampling for Few-Shot Meta-Learning

## Overview

Episodic task sampling is the data pipeline that feeds meta-learning algorithms (MAML, FOMAML,
Reptile) with structured few-shot tasks. Each episode constructs an N-way K-shot classification
problem by sampling a subset of classes from a larger dataset, then drawing a small support
(train) set and a disjoint query (eval) set from those classes. The meta-learner adapts its
parameters on the support set and evaluates generalization on the query set, producing the
outer-loop meta-gradient.

The central engineering constraint is **reproducibility**. Given identical `(global_seed, epoch,
episode_idx)` triples, the sampler must produce byte-identical episodes across runs, machines,
and PyTorch versions. Every source of randomness -- class selection, example selection,
augmentation transforms -- must derive from a deterministic, episode-local random number
generator. Global RNG state must never be consumed or mutated by the sampler.

### Terminology

| Term | Definition |
|---|---|
| **N-way** | Number of classes sampled per episode |
| **K-shot** | Number of support (training) examples per class |
| **Q-query** | Number of query (evaluation) examples per class |
| **Episode** | One N-way K-shot task: support set + query set |
| **Meta-train split** | Set of classes used for training episodes |
| **Meta-val split** | Set of classes used for validation episodes (disjoint from train) |
| **Meta-test split** | Set of classes used for final evaluation episodes (disjoint from train and val) |
| **Class-disjoint** | No class ID appears in more than one split |
| **TaskBatch** | A batch of T episodes collated for vectorized meta-training |

### Design Principles

1. **Deterministic by construction.** Seed every RNG from the episode triple `(seed, epoch, idx)`.
   Never call `torch.rand()`, `random.choice()`, or `numpy.random` without an explicit generator.
2. **Class-disjoint splits.** Split on class identities, not on individual samples. A class that
   appears in meta-train must never appear in meta-val or meta-test.
3. **Pre-computed indices.** Build the `class_to_indices` mapping once at initialization. Episode
   sampling is an index-lookup operation, not a scan over the dataset.
4. **Stateless episodes.** The sampler holds no mutable state between `sample_episode` calls.
   All randomness is derived from the arguments. This makes parallel and out-of-order sampling safe.
5. **Minimal tensor copies.** Return index tensors where possible; defer actual data loading to
   the collation step. For in-memory datasets (Omniglot, mini-ImageNet), direct indexing is
   acceptable.


## EpisodeSampler Contract

The `EpisodeSampler` is the primary interface. It accepts a dataset, a class split, episode
parameters, and a global seed. Its single public method, `sample_episode`, returns an `Episode`
dataclass.

### Constructor

```python
class EpisodeSampler:
    """Deterministic episodic task sampler for few-shot meta-learning.

    Args:
        dataset: Indexable dataset returning (x, y) pairs.
        class_split: Mapping from split name to list of class IDs.
            Example: {'train': [0,1,...,63], 'val': [64,...,79], 'test': [80,...,99]}
        split: Which split to sample from ('train', 'val', or 'test').
        n_way: Number of classes per episode.
        k_shot: Support examples per class.
        q_query: Query examples per class.
        seed: Global seed for reproducibility.
    """

    def __init__(
        self,
        dataset,
        class_split: Dict[str, List[int]],
        split: str = 'train',
        n_way: int = 5,
        k_shot: int = 1,
        q_query: int = 15,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.seed = seed
        self.available_classes = sorted(class_split[split])

        # Pre-compute class-to-index mapping (done once)
        self.class_to_indices = self._build_class_index()

        # Validate: each class must have at least K+Q examples
        for cls_id in self.available_classes:
            n_examples = len(self.class_to_indices[cls_id])
            assert n_examples >= k_shot + q_query, (
                f"Class {cls_id} has {n_examples} examples, "
                f"need at least {k_shot + q_query} (K={k_shot} + Q={q_query})"
            )

    def _build_class_index(self) -> Dict[int, List[int]]:
        """Build mapping from class ID to dataset indices."""
        class_to_indices = defaultdict(list)
        for idx in range(len(self.dataset)):
            _, label = self.dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            class_to_indices[label].append(idx)
        return dict(class_to_indices)
```

### Episode Dataclass

```python
@dataclass
class Episode:
    """A single N-way K-shot episode.

    Attributes:
        support_x: Support set inputs. Shape (N*K, ...) where ... is the
            data shape (e.g., C,H,W for images or D for embeddings).
        support_y: Support set labels. Shape (N*K,) with values in [0, N).
            Labels are remapped to contiguous range starting at 0.
        query_x: Query set inputs. Shape (N*Q, ...).
        query_y: Query set labels. Shape (N*Q,) with values in [0, N).
        class_ids: Original dataset class IDs for the N sampled classes.
            Useful for logging and debugging. Length N.
        episode_id: Tuple (epoch, episode_idx) uniquely identifying this episode.
    """
    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    class_ids: List[int]
    episode_id: Tuple[int, int]
```

**Label remapping.** Original dataset class IDs (e.g., WordNet synset offsets in ImageNet) are
remapped to contiguous integers `[0, N)` within each episode. The mapping is determined by the
order in which classes are selected by `randperm`. Store the original class IDs in `class_ids`
for diagnostics.


## Deterministic Sampling Algorithm

The core algorithm creates a fresh `torch.Generator` per episode, seeded deterministically from
the global seed, epoch number, and episode index. This ensures:

- Identical episodes for the same `(seed, epoch, episode_idx)` regardless of sampling order.
- No cross-contamination between episodes (each has an independent RNG stream).
- Safety under multi-worker DataLoader: workers produce identical results to single-threaded mode.

### Full Implementation

```python
def sample_episode(self, epoch: int, episode_idx: int) -> Episode:
    """Sample a single deterministic episode.

    Args:
        epoch: Current training epoch (0-indexed).
        episode_idx: Episode index within the epoch (0-indexed).

    Returns:
        Episode with support/query sets and metadata.

    Determinism guarantee:
        Calling sample_episode(seed=S, epoch=E, idx=I) always returns
        the same Episode, regardless of call order or parallelism.
    """
    # 1. Create episode-local RNG
    rng = torch.Generator()
    episode_seed = hash((self.seed, epoch, episode_idx)) % (2**63)
    rng.manual_seed(episode_seed)

    # 2. Sample N classes (without replacement) from available classes
    num_available = len(self.available_classes)
    class_perm = torch.randperm(num_available, generator=rng)
    selected_classes = [self.available_classes[i] for i in class_perm[:self.n_way]]

    # 3. For each class, sample K+Q examples (without replacement)
    support_x_list, support_y_list = [], []
    query_x_list, query_y_list = [], []

    for new_label, class_id in enumerate(selected_classes):
        class_indices = self.class_to_indices[class_id]
        num_class_examples = len(class_indices)

        # Permute indices for this class using the same episode RNG
        perm = torch.randperm(num_class_examples, generator=rng)
        selected = [class_indices[perm[i].item()] for i in range(self.k_shot + self.q_query)]

        support_indices = selected[:self.k_shot]
        query_indices = selected[self.k_shot:]

        # Gather data
        for idx in support_indices:
            x, _ = self.dataset[idx]
            support_x_list.append(x)
            support_y_list.append(new_label)

        for idx in query_indices:
            x, _ = self.dataset[idx]
            query_x_list.append(x)
            query_y_list.append(new_label)

    # 4. Stack into tensors
    support_x = torch.stack(support_x_list)      # (N*K, ...)
    support_y = torch.tensor(support_y_list)      # (N*K,)
    query_x = torch.stack(query_x_list)           # (N*Q, ...)
    query_y = torch.tensor(query_y_list)          # (N*Q,)

    # 5. Shuffle support and query sets independently
    support_perm = torch.randperm(len(support_y_list), generator=rng)
    query_perm = torch.randperm(len(query_y_list), generator=rng)
    support_x = support_x[support_perm]
    support_y = support_y[support_perm]
    query_x = query_x[query_perm]
    query_y = query_y[query_perm]

    return Episode(
        support_x=support_x,
        support_y=support_y,
        query_x=query_x,
        query_y=query_y,
        class_ids=selected_classes,
        episode_id=(epoch, episode_idx),
    )
```

### Why `hash((seed, epoch, idx))` Instead of Sequential Seeds

Using `seed + epoch * M + idx` is fragile: overlapping seed ranges between epochs can
produce correlated episodes. Python's `hash()` on a tuple distributes the seed space uniformly
and avoids linear correlations. The `% (2**63)` ensures the seed fits in a signed 64-bit
integer, which is the range accepted by `torch.Generator.manual_seed()`.

**Important:** Python's `hash()` is randomized by default (via `PYTHONHASHSEED`). For
cross-run reproducibility, set `PYTHONHASHSEED=0` in the environment, or replace `hash()` with
a deterministic hash function:

```python
import hashlib
import struct

def deterministic_seed(seed: int, epoch: int, episode_idx: int) -> int:
    """Compute a deterministic episode seed independent of PYTHONHASHSEED."""
    data = struct.pack('>qqq', seed, epoch, episode_idx)
    digest = hashlib.sha256(data).digest()
    return int.from_bytes(digest[:8], 'big') % (2**63)
```

Use this `deterministic_seed` function in production to guarantee reproducibility across
Python invocations regardless of hash randomization settings.


## Class-Disjoint Splits

Meta-learning evaluation is only valid if the meta-test classes were never seen during
meta-training. Splitting must operate on **class identities**, not individual images.

### Constructing Splits

```python
def make_class_splits(
    all_class_ids: List[int],
    train_frac: float = 0.64,
    val_frac: float = 0.16,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Split classes into disjoint train/val/test sets.

    Args:
        all_class_ids: Complete list of class IDs in the dataset.
        train_frac: Fraction of classes for meta-training.
        val_frac: Fraction of classes for meta-validation.
        seed: Seed for the split permutation.

    Returns:
        Dict with 'train', 'val', 'test' keys mapping to class ID lists.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    n = len(all_class_ids)
    perm = torch.randperm(n, generator=rng)
    sorted_ids = sorted(all_class_ids)
    shuffled = [sorted_ids[perm[i].item()] for i in range(n)]

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    return {
        'train': sorted(shuffled[:n_train]),
        'val': sorted(shuffled[n_train:n_train + n_val]),
        'test': sorted(shuffled[n_train + n_val:]),
    }
```

### Standard Benchmark Splits

For established benchmarks, use the canonical splits rather than generating new ones.
This ensures comparability with published results.

**Omniglot (Lake et al.):**
- Background set: 30 alphabets, 964 characters -- use for meta-train.
- Evaluation set: 20 alphabets, 659 characters -- split further into meta-val and meta-test,
  or use entirely for meta-test.
- The split is at the alphabet level, which naturally produces class-disjoint sets since
  characters within an alphabet are distinct.

**mini-ImageNet (Ravi & Larochelle, 2017):**
- 64 train classes / 16 validation classes / 20 test classes.
- The exact WordNet IDs (wnids) for each split must be loaded from the standard CSV files
  distributed with the benchmark. Do not re-derive the split.

```python
MINI_IMAGENET_SPLITS = {
    'train': [  # 64 classes
        'n01532829', 'n01558993', 'n01704323', 'n01749939', 'n01770081',
        'n01843383', 'n01855672', 'n01910747', 'n01930112', 'n01981276',
        'n02074367', 'n02089867', 'n02091244', 'n02091831', 'n02099601',
        'n02101006', 'n02105505', 'n02108089', 'n02108551', 'n02108915',
        'n02110063', 'n02110341', 'n02111277', 'n02113712', 'n02114548',
        'n02116738', 'n02120079', 'n02129165', 'n02138441', 'n02165456',
        'n02174001', 'n02219486', 'n02443484', 'n02457408', 'n02606052',
        'n02687172', 'n02747177', 'n02795169', 'n02823428', 'n02871525',
        'n02950826', 'n02966193', 'n02971356', 'n02981792', 'n03017168',
        'n03047690', 'n03062245', 'n03075370', 'n03127925', 'n03146219',
        'n03207743', 'n03220513', 'n03272010', 'n03337140', 'n03347037',
        'n03400231', 'n03417042', 'n03476684', 'n03527444', 'n03535780',
        'n03544143', 'n03584254', 'n03676483', 'n03770439',
    ],
    'val': [  # 16 classes
        'n01855032', 'n02108000', 'n02110958', 'n02111500', 'n02120623',
        'n02165105', 'n02457823', 'n02606384', 'n02823750', 'n02950812',
        'n03017168', 'n03476991', 'n03530642', 'n03584829', 'n03770679',
        'n03838899',
    ],
    'test': [  # 20 classes
        'n01532828', 'n01558594', 'n01704401', 'n01749401', 'n01770393',
        'n01843065', 'n01910546', 'n02074604', 'n02091032', 'n02099712',
        'n02105056', 'n02108422', 'n02110806', 'n02113186', 'n02116910',
        'n02129604', 'n02138251', 'n02174659', 'n02443114', 'n02606541',
    ],
}
```

**Validation rule:** Assert at initialization that the intersection of any two splits is empty:

```python
train_set = set(class_split['train'])
val_set = set(class_split['val'])
test_set = set(class_split['test'])
assert train_set.isdisjoint(val_set), "Train/val class overlap detected"
assert train_set.isdisjoint(test_set), "Train/test class overlap detected"
assert val_set.isdisjoint(test_set), "Val/test class overlap detected"
```


## Omniglot Dataset

### Dataset Statistics

| Property | Value |
|---|---|
| Total characters | 1,623 |
| Alphabets | 50 |
| Examples per character | 20 |
| Image size (original) | 105 x 105, grayscale |
| Image size (standard resize) | 28 x 28, grayscale |
| Background alphabets | 30 (964 characters) |
| Evaluation alphabets | 20 (659 characters) |
| With rotation augmentation | 6,492 effective classes |

### Loading Pattern

```python
class OmniglotDataset:
    """Omniglot dataset for few-shot meta-learning.

    Args:
        root: Path to omniglot directory (containing images_background/ and images_evaluation/).
        split: 'background' or 'evaluation'.
        resize: Target image size (default 28).
        use_rotations: Multiply classes by 4 via 90-degree rotations (default True).
        transform: Optional additional transform applied after resize.
    """

    def __init__(
        self,
        root: str,
        split: str = 'background',
        resize: int = 28,
        use_rotations: bool = True,
        transform: Optional[Callable] = None,
    ):
        self.resize = resize
        self.use_rotations = use_rotations
        self.transform = transform

        # Determine directory
        if split == 'background':
            data_dir = os.path.join(root, 'images_background')
        elif split == 'evaluation':
            data_dir = os.path.join(root, 'images_evaluation')
        else:
            raise ValueError(f"Unknown split: {split}. Use 'background' or 'evaluation'.")

        # Walk directory: root/alphabet/character/image.png
        self.samples = []  # List of (image_path, class_id)
        self.class_to_indices = defaultdict(list)
        class_id = 0

        for alphabet in sorted(os.listdir(data_dir)):
            alphabet_dir = os.path.join(data_dir, alphabet)
            if not os.path.isdir(alphabet_dir):
                continue
            for character in sorted(os.listdir(alphabet_dir)):
                char_dir = os.path.join(alphabet_dir, character)
                if not os.path.isdir(char_dir):
                    continue

                image_files = sorted([
                    f for f in os.listdir(char_dir) if f.endswith('.png')
                ])

                if use_rotations:
                    # Each rotation produces a distinct class
                    for rot_idx, angle in enumerate([0, 90, 180, 270]):
                        rot_class_id = class_id + rot_idx
                        for img_file in image_files:
                            img_path = os.path.join(char_dir, img_file)
                            sample_idx = len(self.samples)
                            self.samples.append((img_path, rot_class_id, angle))
                            self.class_to_indices[rot_class_id].append(sample_idx)
                    class_id += 4
                else:
                    for img_file in image_files:
                        img_path = os.path.join(char_dir, img_file)
                        sample_idx = len(self.samples)
                        self.samples.append((img_path, class_id, 0))
                        self.class_to_indices[class_id].append(sample_idx)
                    class_id += 1

        self.num_classes = class_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_id, angle = self.samples[idx]
        img = Image.open(img_path).convert('L')  # Grayscale

        # Resize
        img = img.resize((self.resize, self.resize), Image.LANCZOS)

        # Apply deterministic rotation (keyed to class, not random)
        if angle != 0:
            img = img.rotate(angle)

        # Convert to tensor: (1, H, W) in [0, 1]
        img_tensor = torch.from_numpy(
            np.array(img, dtype=np.float32) / 255.0
        ).unsqueeze(0)

        # Optional additional transform
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        return img_tensor, class_id
```

### Rotation Augmentation Details

Rotation augmentation in Omniglot is **not** a random augmentation applied during training. It is
a deterministic class-expansion strategy:

- Character `C` at rotation 0 degrees becomes class `4*C + 0`.
- Character `C` at rotation 90 degrees becomes class `4*C + 1`.
- Character `C` at rotation 180 degrees becomes class `4*C + 2`.
- Character `C` at rotation 270 degrees becomes class `4*C + 3`.

Each rotated variant is a **separate class** with its own 20 examples. The rotation angle is
fixed per sample at dataset construction time, not sampled per episode. This is critical for
reproducibility: the same class ID always refers to the same character at the same rotation.

### Omniglot Class Split for Meta-Learning

```python
def omniglot_class_split(dataset: OmniglotDataset) -> Dict[str, List[int]]:
    """Return class-disjoint splits for Omniglot.

    Background alphabets -> meta-train (964 chars * 4 rotations = 3,856 classes)
    Evaluation alphabets -> split into meta-val and meta-test

    For standard benchmarks, load background and evaluation as separate
    OmniglotDataset instances and use each as a complete split.
    """
    # When using the standard split, all classes in the background dataset
    # are meta-train, and all classes in the evaluation dataset are meta-test.
    # For a train/val/test split, sub-split evaluation alphabets:
    # 10 alphabets -> val, 10 alphabets -> test.
    all_classes = list(range(dataset.num_classes))
    return {'train': all_classes}  # Full split used for the given dataset instance
```


## mini-ImageNet Dataset

### Dataset Statistics

| Property | Value |
|---|---|
| Total classes | 100 |
| Images per class | 600 |
| Total images | 60,000 |
| Image size | 84 x 84, RGB |
| Train classes (Ravi & Larochelle) | 64 |
| Validation classes | 16 |
| Test classes | 20 |

### Loading Pattern

```python
class MiniImageNetDataset:
    """mini-ImageNet dataset for few-shot meta-learning.

    Expects pre-processed images stored as:
        root/
            images/
                n01532829/
                    n01532829_00001.jpg
                    ...
                ...
        or a single numpy file with all images.

    Args:
        root: Path to mini-ImageNet directory.
        split: 'train', 'val', or 'test'.
        image_size: Target image size (default 84).
        transform: Optional transform for training augmentation.
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        image_size: int = 84,
        transform: Optional[Callable] = None,
    ):
        self.image_size = image_size
        self.transform = transform
        self.split = split

        # Load split class IDs
        split_classes = MINI_IMAGENET_SPLITS[split]

        # Build samples list and class-to-indices mapping
        self.samples = []
        self.class_to_indices = defaultdict(list)

        # Map wnid -> contiguous integer for this split
        self.wnid_to_label = {wnid: i for i, wnid in enumerate(sorted(split_classes))}

        images_dir = os.path.join(root, 'images')
        for wnid in sorted(split_classes):
            class_dir = os.path.join(images_dir, wnid)
            label = self.wnid_to_label[wnid]

            for img_file in sorted(os.listdir(class_dir)):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                img_path = os.path.join(class_dir, img_file)
                sample_idx = len(self.samples)
                self.samples.append((img_path, label))
                self.class_to_indices[label].append(sample_idx)

        self.num_classes = len(split_classes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)

        img_tensor = torch.from_numpy(
            np.array(img, dtype=np.float32) / 255.0
        ).permute(2, 0, 1)  # (3, H, W)

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label
```

### Augmentation for mini-ImageNet

Apply augmentations **only during meta-training episodes**, not during meta-validation or
meta-test. Standard augmentations:

| Augmentation | Parameters | Notes |
|---|---|---|
| Random crop | 84 x 84 from 92 x 92 padded | Pad with reflection, then crop |
| Horizontal flip | p=0.5 | Applied per-image, not per-class |
| Color jitter | brightness=0.4, contrast=0.4, saturation=0.4 | Applied per-image |
| Normalization | ImageNet mean/std | `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` |

**Deterministic augmentation:** When augmentations are applied inside `sample_episode`, derive
the augmentation RNG from the episode seed. When augmentations are applied in the dataset
`__getitem__`, they are non-deterministic by default. Prefer the episode-seeded approach for
reproducibility:

```python
def apply_augmentation(img_tensor: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Apply deterministic augmentations using episode-local RNG."""
    # Random horizontal flip
    if torch.rand(1, generator=rng).item() > 0.5:
        img_tensor = img_tensor.flip(-1)  # Flip width dimension

    # Random crop (pad then crop)
    pad = 4
    padded = F.pad(img_tensor, [pad]*4, mode='reflect')
    h, w = img_tensor.shape[-2:]
    top = torch.randint(0, 2*pad, (1,), generator=rng).item()
    left = torch.randint(0, 2*pad, (1,), generator=rng).item()
    img_tensor = padded[..., top:top+h, left:left+w]

    return img_tensor
```


## Reproducibility Rules

### Rule 1: Seed Everything from the Episode Triple

Every random operation inside `sample_episode` must use the episode-local `torch.Generator`.
Never read from or write to the global `torch.default_generator`.

```python
# CORRECT: episode-local RNG
rng = torch.Generator()
rng.manual_seed(deterministic_seed(self.seed, epoch, episode_idx))
perm = torch.randperm(n, generator=rng)

# INCORRECT: consumes global RNG state
perm = torch.randperm(n)  # Non-reproducible across calls
```

### Rule 2: Deterministic Augmentations

If augmentations are applied within the episode sampler, derive their randomness from the
same episode RNG (called after the class/example selection calls, so the sequence is fixed).
If augmentations are applied in the dataset `__getitem__`, they break reproducibility unless
the dataset also receives an episode-local seed.

Preferred approach: apply augmentations in `sample_episode` after gathering raw tensors, using
the episode `rng`. This keeps all randomness in one place.

### Rule 3: Pre-Compute Class-to-Index Mappings

Build `class_to_indices` once in `__init__`, not per episode. This avoids both redundant
computation and non-determinism from filesystem ordering differences.

```python
# In __init__:
self.class_to_indices = {}
for idx in range(len(dataset)):
    _, label = dataset[idx]
    self.class_to_indices.setdefault(label, []).append(idx)

# Sort indices for determinism (filesystem order may vary)
for cls_id in self.class_to_indices:
    self.class_to_indices[cls_id].sort()
```

### Rule 4: Log Episode Metadata

Log the `class_ids` and `episode_id` for every episode during training. This enables:

- Reproduction of a specific episode for debugging.
- Verification that the same seed produces the same episode sequence.
- Analysis of class sampling frequency over training.

```python
# In the training loop:
episode = sampler.sample_episode(epoch, idx)
logger.debug(
    f"Episode ({epoch}, {idx}): classes={episode.class_ids}, "
    f"support_shape={episode.support_x.shape}"
)
```

### Rule 5: Use `torch.Generator`, Not `numpy.random`

`torch.Generator` behavior is consistent across platforms (CPU). NumPy's `RandomState` can
differ between versions and platforms. For maximum portability, keep all sampling in PyTorch.

```python
# CORRECT: torch.Generator for cross-platform reproducibility
rng = torch.Generator()
rng.manual_seed(seed)
perm = torch.randperm(n, generator=rng)

# AVOID: numpy may differ across platforms/versions
rng_np = np.random.RandomState(seed)
perm = rng_np.permutation(n)
```

### Rule 6: Worker Safety in DataLoader

When using `num_workers > 0`, each worker process gets its own copy of the sampler. Because
`sample_episode` is stateless (all randomness comes from the arguments), workers automatically
produce correct results. However, verify that:

- The dataset object is safe for multi-process access (no shared mutable state).
- File handles are opened per-access, not cached at init.
- The worker init function does not modify global RNG state.

```python
def worker_init_fn(worker_id):
    """Initialize worker with deterministic seed. Do not touch global RNG."""
    # No-op: EpisodeSampler handles its own RNG per episode.
    # Only set this if the dataset uses random transforms outside the sampler.
    pass
```

### Rule 7: Environment Variables for Full Determinism

Set the following environment variables for bit-exact reproducibility:

```bash
PYTHONHASHSEED=0          # Disable hash randomization
CUBLAS_WORKSPACE_CONFIG=:4096:8  # Deterministic cuBLAS
```

And in code:

```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Note: `torch.use_deterministic_algorithms(True)` may raise errors for operations without
deterministic implementations. Use `torch.use_deterministic_algorithms(True, warn_only=True)`
during development and enforce strict mode in CI.


## TaskBatch for Meta-Training

A `TaskBatch` collates multiple episodes into a single batched structure for vectorized
meta-training. This is the input format expected by `meta_loss()` in the algorithm layer.

### TaskBatch Dataclass

```python
@dataclass
class TaskBatch:
    """Batch of T episodes for vectorized meta-training.

    Attributes:
        episodes: List of T Episode instances.
    """
    episodes: List[Episode]

    def __len__(self) -> int:
        return len(self.episodes)

    @property
    def support(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stacked support sets.

        Returns:
            support_x: (T, N*K, ...) stacked support inputs.
            support_y: (T, N*K) stacked support labels.
        """
        sx = torch.stack([ep.support_x for ep in self.episodes])
        sy = torch.stack([ep.support_y for ep in self.episodes])
        return sx, sy

    @property
    def query(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stacked query sets.

        Returns:
            query_x: (T, N*Q, ...) stacked query inputs.
            query_y: (T, N*Q) stacked query labels.
        """
        qx = torch.stack([ep.query_x for ep in self.episodes])
        qy = torch.stack([ep.query_y for ep in self.episodes])
        return qx, qy

    @property
    def class_ids(self) -> List[List[int]]:
        """Original class IDs for each episode. Length T, each inner list length N."""
        return [ep.class_ids for ep in self.episodes]
```

### Collation Function

Use a collation function to build `TaskBatch` instances from a sequence of episodes:

```python
def collate_episodes(episodes: List[Episode]) -> TaskBatch:
    """Collate a list of episodes into a TaskBatch.

    All episodes must have the same N, K, Q and data shapes.
    """
    if len(episodes) == 0:
        raise ValueError("Cannot collate empty episode list")

    # Validate uniform shapes
    ref = episodes[0]
    for ep in episodes[1:]:
        assert ep.support_x.shape == ref.support_x.shape, (
            f"Support shape mismatch: {ep.support_x.shape} vs {ref.support_x.shape}"
        )
        assert ep.query_x.shape == ref.query_x.shape, (
            f"Query shape mismatch: {ep.query_x.shape} vs {ref.query_x.shape}"
        )

    return TaskBatch(episodes=episodes)
```

### Memory Considerations

For large tasks (e.g., 84x84 RGB images at 5-way 15-query), a single `TaskBatch` of T=4
episodes occupies:

- Support: `4 * 5 * 1 * 3 * 84 * 84 * 4 bytes = ~3.4 MB` (5-way 1-shot)
- Query: `4 * 5 * 15 * 3 * 84 * 84 * 4 bytes = ~50.8 MB` (5-way 15-query)

This fits comfortably in GPU memory. For larger batch sizes or image resolutions, consider
lazy loading or pin-memory with async transfer.


## Custom Dataset Integration

To use the episode sampler with a non-standard dataset, provide a dataset that supports
`__len__` and `__getitem__` returning `(data_tensor, label_int)`, plus a class split.

### Adapter Pattern

```python
class CustomFewShotDataset:
    """Adapter for arbitrary datasets to work with EpisodeSampler.

    Args:
        data: Tensor of shape (N_samples, ...) or list of tensors.
        labels: Tensor or list of integer labels, length N_samples.
        class_split: Dict mapping split names to lists of class IDs.
            If None, all classes go to 'train'.
    """

    def __init__(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        labels: Union[torch.Tensor, List[int]],
        class_split: Optional[Dict[str, List[int]]] = None,
    ):
        if isinstance(data, list):
            self.data = data
        else:
            self.data = data

        if isinstance(labels, torch.Tensor):
            self.labels = labels.tolist()
        else:
            self.labels = list(labels)

        assert len(self.data) == len(self.labels)

        # Build class-to-indices mapping
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_to_indices[label].append(idx)

        # Class split
        if class_split is None:
            all_classes = sorted(set(self.labels))
            self.class_split = {'train': all_classes}
        else:
            self.class_split = class_split

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if isinstance(self.data, torch.Tensor):
            return self.data[idx], self.labels[idx]
        else:
            return self.data[idx], self.labels[idx]
```

### Requirements for Custom Datasets

| Requirement | Details |
|---|---|
| Minimum examples per class | At least `K + Q` examples for every class in the split |
| Label type | Integer labels (not strings, not one-hot) |
| Data shape consistency | All `__getitem__` calls must return tensors of the same shape |
| Index stability | `__getitem__(i)` must return the same `(x, y)` across calls |
| Thread safety | Safe for concurrent access from multiple DataLoader workers |


## DataLoader Integration

### EpisodeDataLoader

Wrap the `EpisodeSampler` in a DataLoader-like iterator for integration with standard
training loops:

```python
class EpisodeDataLoader:
    """DataLoader-style iterator over episodes.

    Args:
        sampler: EpisodeSampler instance.
        episodes_per_epoch: Number of episodes per epoch.
        batch_size: Number of episodes per TaskBatch.
        num_workers: Number of parallel workers (0 = main process).
        pin_memory: Pin TaskBatch tensors to GPU-friendly memory.
    """

    def __init__(
        self,
        sampler: EpisodeSampler,
        episodes_per_epoch: int = 600,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        self.sampler = sampler
        self.episodes_per_epoch = episodes_per_epoch
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.epoch = 0

    def __len__(self) -> int:
        """Number of TaskBatches per epoch."""
        return (self.episodes_per_epoch + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """Yield TaskBatch instances for one epoch."""
        batch = []
        for idx in range(self.episodes_per_epoch):
            episode = self.sampler.sample_episode(self.epoch, idx)
            batch.append(episode)

            if len(batch) == self.batch_size:
                task_batch = collate_episodes(batch)
                if self.pin_memory:
                    task_batch = self._pin(task_batch)
                yield task_batch
                batch = []

        # Yield remaining episodes (partial batch)
        if batch:
            task_batch = collate_episodes(batch)
            if self.pin_memory:
                task_batch = self._pin(task_batch)
            yield task_batch

    def set_epoch(self, epoch: int):
        """Set the current epoch for deterministic sampling."""
        self.epoch = epoch

    def _pin(self, task_batch: TaskBatch) -> TaskBatch:
        """Pin episode tensors to page-locked memory."""
        for ep in task_batch.episodes:
            ep.support_x = ep.support_x.pin_memory()
            ep.support_y = ep.support_y.pin_memory()
            ep.query_x = ep.query_x.pin_memory()
            ep.query_y = ep.query_y.pin_memory()
        return task_batch
```

### Integration with Standard PyTorch DataLoader

For workloads where episode construction is I/O-bound (loading images from disk), use
`torch.utils.data.DataLoader` with a custom dataset that returns episodes:

```python
class EpisodeDataset(torch.utils.data.Dataset):
    """Wrap EpisodeSampler as a map-style Dataset for use with DataLoader."""

    def __init__(self, sampler: EpisodeSampler, episodes_per_epoch: int, epoch: int = 0):
        self.sampler = sampler
        self.episodes_per_epoch = episodes_per_epoch
        self.epoch = epoch

    def __len__(self):
        return self.episodes_per_epoch

    def __getitem__(self, idx):
        return self.sampler.sample_episode(self.epoch, idx)

    def set_epoch(self, epoch: int):
        self.epoch = epoch


# Usage with torch DataLoader:
episode_ds = EpisodeDataset(sampler, episodes_per_epoch=600)
loader = torch.utils.data.DataLoader(
    episode_ds,
    batch_size=4,
    collate_fn=collate_episodes,
    num_workers=4,
    worker_init_fn=worker_init_fn,
    pin_memory=True,
)
```

The `worker_init_fn` can be a no-op because `sample_episode` uses episode-local RNG. The
DataLoader's built-in index distribution handles which worker samples which episode index.


## Testing Patterns

### Test 1: Determinism

Verify that the same `(seed, epoch, episode_idx)` always produces an identical episode.

```python
def test_episode_determinism():
    """Same seed + epoch + idx must produce identical episodes."""
    dataset = make_test_dataset()  # Small synthetic dataset
    split = {'train': list(range(20))}
    sampler = EpisodeSampler(dataset, split, n_way=5, k_shot=1, q_query=5, seed=42)

    ep1 = sampler.sample_episode(epoch=0, episode_idx=0)
    ep2 = sampler.sample_episode(epoch=0, episode_idx=0)

    assert torch.equal(ep1.support_x, ep2.support_x), "Support X mismatch"
    assert torch.equal(ep1.support_y, ep2.support_y), "Support Y mismatch"
    assert torch.equal(ep1.query_x, ep2.query_x), "Query X mismatch"
    assert torch.equal(ep1.query_y, ep2.query_y), "Query Y mismatch"
    assert ep1.class_ids == ep2.class_ids, "Class IDs mismatch"
```

### Test 2: Different Episodes Are Different

```python
def test_episode_variation():
    """Different episode indices must (almost certainly) produce different episodes."""
    sampler = EpisodeSampler(dataset, split, n_way=5, k_shot=1, q_query=5, seed=42)

    ep1 = sampler.sample_episode(epoch=0, episode_idx=0)
    ep2 = sampler.sample_episode(epoch=0, episode_idx=1)

    # At least one of class_ids or support data should differ
    assert ep1.class_ids != ep2.class_ids or not torch.equal(ep1.support_x, ep2.support_x)
```

### Test 3: Class-Disjoint Splits

```python
def test_class_disjoint_splits():
    """No class appears in more than one split."""
    split = make_class_splits(list(range(100)), train_frac=0.64, val_frac=0.16, seed=0)

    train_set = set(split['train'])
    val_set = set(split['val'])
    test_set = set(split['test'])

    assert train_set.isdisjoint(val_set), f"Overlap: {train_set & val_set}"
    assert train_set.isdisjoint(test_set), f"Overlap: {train_set & test_set}"
    assert val_set.isdisjoint(test_set), f"Overlap: {val_set & test_set}"
    assert len(train_set) + len(val_set) + len(test_set) == 100
```

### Test 4: Coverage

```python
def test_class_coverage():
    """Over many episodes, all classes in the split are eventually sampled."""
    sampler = EpisodeSampler(dataset, split, n_way=5, k_shot=1, q_query=5, seed=42)
    available = set(sampler.available_classes)
    sampled = set()

    for idx in range(500):
        ep = sampler.sample_episode(epoch=0, episode_idx=idx)
        sampled.update(ep.class_ids)

    missing = available - sampled
    assert len(missing) == 0, f"Classes never sampled after 500 episodes: {missing}"
```

### Test 5: Balance

```python
def test_episode_balance():
    """Each episode has exactly N*K support and N*Q query examples."""
    sampler = EpisodeSampler(dataset, split, n_way=5, k_shot=1, q_query=15, seed=42)

    for idx in range(50):
        ep = sampler.sample_episode(epoch=0, episode_idx=idx)

        assert ep.support_x.shape[0] == 5 * 1, f"Expected 5 support, got {ep.support_x.shape[0]}"
        assert ep.query_x.shape[0] == 5 * 15, f"Expected 75 query, got {ep.query_x.shape[0]}"
        assert ep.support_y.shape[0] == 5 * 1
        assert ep.query_y.shape[0] == 5 * 15

        # Check label balance: each label [0, N) appears exactly K or Q times
        for label in range(5):
            s_count = (ep.support_y == label).sum().item()
            q_count = (ep.query_y == label).sum().item()
            assert s_count == 1, f"Label {label}: expected 1 support, got {s_count}"
            assert q_count == 15, f"Label {label}: expected 15 query, got {q_count}"
```

### Test 6: Support-Query Disjointness

```python
def test_support_query_disjoint():
    """No example appears in both the support and query sets of the same episode."""
    sampler = EpisodeSampler(dataset, split, n_way=5, k_shot=5, q_query=5, seed=42)

    for idx in range(50):
        ep = sampler.sample_episode(epoch=0, episode_idx=idx)

        # Compare raw data: no row in support_x should appear in query_x
        for i in range(ep.support_x.shape[0]):
            for j in range(ep.query_x.shape[0]):
                assert not torch.equal(ep.support_x[i], ep.query_x[j]), (
                    f"Episode {idx}: support[{i}] == query[{j}]"
                )
```

### Test 7: Cross-Epoch Determinism

```python
def test_cross_epoch_determinism():
    """Same episode_idx in different epochs produces different episodes."""
    sampler = EpisodeSampler(dataset, split, n_way=5, k_shot=1, q_query=5, seed=42)

    ep_0_0 = sampler.sample_episode(epoch=0, episode_idx=0)
    ep_1_0 = sampler.sample_episode(epoch=1, episode_idx=0)

    # Different epochs, same idx -> (almost certainly) different episodes
    assert ep_0_0.class_ids != ep_1_0.class_ids or not torch.equal(
        ep_0_0.support_x, ep_1_0.support_x
    )
```

### Test 8: Serialization Round-Trip

```python
def test_episode_serialization():
    """Episode can be saved and loaded with exact fidelity."""
    ep = sampler.sample_episode(epoch=0, episode_idx=0)

    # Save as individual tensors (safe serialization)
    state = {
        'support_x': ep.support_x,
        'support_y': ep.support_y,
        'query_x': ep.query_x,
        'query_y': ep.query_y,
        'class_ids': ep.class_ids,
        'episode_id': ep.episode_id,
    }

    # Use safetensors or torch.save with weights_only=True for loading
    torch.save(state, '/tmp/episode_test.pt')
    loaded = torch.load('/tmp/episode_test.pt', weights_only=False)

    assert torch.equal(loaded['support_x'], ep.support_x)
    assert torch.equal(loaded['query_x'], ep.query_x)
    assert loaded['class_ids'] == ep.class_ids
```


## Appendix A: Dataset Statistics Quick Reference

| Dataset | Classes | Examples/Class | Image Size | Channels | Split Sizes |
|---|---|---|---|---|---|
| Omniglot (no rot) | 1,623 | 20 | 28x28 | 1 (grayscale) | 964 / 659 |
| Omniglot (with rot) | 6,492 | 20 | 28x28 | 1 (grayscale) | 3,856 / 2,636 |
| mini-ImageNet | 100 | 600 | 84x84 | 3 (RGB) | 64 / 16 / 20 |
| tiered-ImageNet | 608 | ~1,281 avg | 84x84 | 3 (RGB) | 351 / 97 / 160 |
| CIFAR-FS | 100 | 600 | 32x32 | 3 (RGB) | 64 / 16 / 20 |
| FC100 | 100 | 600 | 32x32 | 3 (RGB) | 60 / 20 / 20 |

tiered-ImageNet and CIFAR-FS / FC100 are included for reference. The EpisodeSampler works
with any dataset that satisfies the `CustomFewShotDataset` interface.


## Appendix B: Common Configurations

### 5-Way 1-Shot (Standard Few-Shot)

```python
EpisodeConfig(
    n_way=5,
    k_shot=1,
    q_query=15,
    episodes_per_epoch=600,
)
```

Memory per episode (Omniglot 28x28):
- Support: `5 * 1 * 1 * 28 * 28 * 4 = 15.7 KB`
- Query: `5 * 15 * 1 * 28 * 28 * 4 = 235 KB`

### 5-Way 5-Shot

```python
EpisodeConfig(
    n_way=5,
    k_shot=5,
    q_query=15,
    episodes_per_epoch=600,
)
```

Memory per episode (mini-ImageNet 84x84 RGB):
- Support: `5 * 5 * 3 * 84 * 84 * 4 = 2.1 MB`
- Query: `5 * 15 * 3 * 84 * 84 * 4 = 6.4 MB`

### 20-Way 1-Shot (Omniglot Standard)

```python
EpisodeConfig(
    n_way=20,
    k_shot=1,
    q_query=5,
    episodes_per_epoch=1000,
)
```

20-way is standard for Omniglot because the large number of classes (6,492 with rotations)
supports wider classification tasks. Use 5-way for mini-ImageNet where fewer classes are available
per split.


## Appendix C: Performance Optimization

### Pre-Loading to Memory

For datasets that fit in memory (Omniglot at ~50 MB, mini-ImageNet at ~2.8 GB), load all images
into a single tensor at initialization:

```python
class InMemoryDataset:
    def __init__(self, dataset):
        all_x, all_y = [], []
        for idx in range(len(dataset)):
            x, y = dataset[idx]
            all_x.append(x)
            all_y.append(y)
        self.data = torch.stack(all_x)    # (N, C, H, W)
        self.labels = torch.tensor(all_y)  # (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx].item()
```

This eliminates per-sample I/O during episode construction and makes `sample_episode` purely
a tensor-indexing operation.

### Vectorized Episode Construction

When the dataset is in memory, replace the per-sample loop with batch indexing:

```python
def sample_episode_vectorized(self, epoch, episode_idx):
    rng = torch.Generator()
    rng.manual_seed(deterministic_seed(self.seed, epoch, episode_idx))

    class_perm = torch.randperm(len(self.available_classes), generator=rng)
    selected = [self.available_classes[i] for i in class_perm[:self.n_way]]

    all_support_idx, all_query_idx = [], []
    support_labels, query_labels = [], []

    for new_label, cls_id in enumerate(selected):
        indices = torch.tensor(self.class_to_indices[cls_id])
        perm = torch.randperm(len(indices), generator=rng)
        chosen = indices[perm[:self.k_shot + self.q_query]]

        all_support_idx.append(chosen[:self.k_shot])
        all_query_idx.append(chosen[self.k_shot:])
        support_labels.extend([new_label] * self.k_shot)
        query_labels.extend([new_label] * self.q_query)

    support_idx = torch.cat(all_support_idx)
    query_idx = torch.cat(all_query_idx)

    # Single batch index into the in-memory tensor
    support_x = self.dataset.data[support_idx]
    query_x = self.dataset.data[query_idx]

    # Shuffle
    s_perm = torch.randperm(len(support_labels), generator=rng)
    q_perm = torch.randperm(len(query_labels), generator=rng)

    return Episode(
        support_x=support_x[s_perm],
        support_y=torch.tensor(support_labels)[s_perm],
        query_x=query_x[q_perm],
        query_y=torch.tensor(query_labels)[q_perm],
        class_ids=selected,
        episode_id=(epoch, episode_idx),
    )
```

### Throughput Targets

| Configuration | Target | Notes |
|---|---|---|
| Omniglot 5-way 1-shot, in-memory | > 5,000 episodes/sec | CPU, single thread |
| Omniglot 20-way 1-shot, in-memory | > 2,000 episodes/sec | CPU, single thread |
| mini-ImageNet 5-way 5-shot, in-memory | > 1,000 episodes/sec | CPU, single thread |
| mini-ImageNet 5-way 5-shot, disk | > 100 episodes/sec | 4 DataLoader workers, SSD |

Measure throughput with `scripts/meta_benchmark.py` to verify the implementation meets these
targets. Episode construction should never be the training bottleneck; the inner-loop forward
and backward passes dominate wall time.


## Appendix D: Integration with BrainAI Meta-Learning Pipeline

In the BrainAI cognitive pipeline, episodic sampling feeds Phase 7 (Meta-Learning). The
`EpisodeSampler` produces episodes that are consumed by the meta-training loop in
`scripts/train_phase7.py` (or via `train_full_pipeline.py --start-phase 7`).

### Configuration Mapping

The `EpisodeConfig` dataclass in `brain_ai/config.py` maps directly to `EpisodeSampler`
constructor arguments:

```python
from brain_ai.config import BrainAIConfig

config = BrainAIConfig.dev()
episode_cfg = config.meta.episode  # EpisodeConfig instance

sampler = EpisodeSampler(
    dataset=dataset,
    class_split=class_split,
    split='train',
    n_way=episode_cfg.n_way,
    k_shot=episode_cfg.k_shot,
    q_query=episode_cfg.q_query,
    seed=config.seed,
)
```

### Episode Flow Through the Pipeline

```
EpisodeSampler.sample_episode()
    -> Episode
    -> collate_episodes()
    -> TaskBatch
    -> meta_loss(params, task_batch, algo=config.meta.algo)
    -> MetaOutput (loss, metrics, inner_logs, adapted_params)
    -> outer_optimizer.step()
```

The sampler is agnostic to the meta-learning algorithm. MAML, FOMAML, and Reptile all consume
the same `TaskBatch` structure. Algorithm-specific behavior (second-order gradients, weight
interpolation) is handled entirely in the `meta_loss` function documented in
`references/algorithm-variants.md`.
