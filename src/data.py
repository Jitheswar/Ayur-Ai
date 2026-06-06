"""Dataset and dataloader construction for leaf-image classification.

Expected layout (one folder per plant class, matching the dataset's names)::

    data/raw/
        Ocimum Tenuiflorum (Tulsi)/  img001.jpg ...
        Azadirachta Indica (Neem)/   img001.jpg ...
        ...

The single ``raw`` folder is split into train / val / test on the fly using
the ratios in ``config.yaml`` (a fixed seed keeps the split reproducible).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .config import ROOT

# ImageNet statistics -- correct because we use ImageNet-pretrained backbones.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int):
    """Return (train_transform, eval_transform)."""
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def inference_transform(image_size: int):
    """The transform used for single-image prediction (same as eval)."""
    return build_transforms(image_size)[1]


def _stratified_split(targets, val_split, test_split, seed):
    """Return (train_idx, val_idx, test_idx) keeping class balance per split."""
    import collections
    import random

    rng = random.Random(seed)
    by_class: dict[int, list[int]] = collections.defaultdict(list)
    for idx, label in enumerate(targets):
        by_class[label].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for label, idxs in by_class.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(round(n * test_split))
        n_val = int(round(n * val_split))
        # Guarantee at least one training sample per class.
        n_val = min(n_val, max(0, n - n_test - 1))
        test_idx += idxs[:n_test]
        val_idx += idxs[n_test : n_test + n_val]
        train_idx += idxs[n_test + n_val :]
    return train_idx, val_idx, test_idx


class _TransformedSubset(Dataset):
    """A view over an ImageFolder that applies its own transform.

    Implemented as a plain Dataset (not ``torch.utils.data.Subset``) so each
    split can use a different transform without tripping Subset's stricter
    ``__getitems__`` contract in recent torch versions.
    """

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = list(indices)
        self._transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        # ImageFolder loads via its own .loader; bypass its transform here.
        actual = self.indices[i]
        path, target = self.dataset.samples[actual]
        sample = self.dataset.loader(path)
        if self._transform is not None:
            sample = self._transform(sample)
        return sample, target


def _component_stratified_split(samples, val_split, test_split, seed, threshold=8):
    """Like _stratified_split, but groups near-duplicate images (pHash Hamming ≤ threshold)
    into components and assigns each component entirely to one split.

    This prevents cross-split leakage caused by near-identical photos of the same leaf
    appearing in both train and test.
    """
    import collections
    import random

    try:
        import imagehash
        from PIL import Image as PILImage
    except ImportError:
        raise ImportError("imagehash is required for deduplicate=true. Run: pip install imagehash")

    # ── 1. Compute pHash for every image (disk-cached; the dataset is static) ──
    # Caching makes repeated splits (e.g. the multi-seed evaluator) skip the
    # ~30 s hashing pass — the hash depends only on the image file, not the seed.
    import json

    cache_path = ROOT / "experiments" / ".phash_cache.json"
    try:
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    except Exception:
        cache = {}

    hashes: dict[int, object] = {}
    dirty = False
    for i, (path, _) in enumerate(samples):
        key = str(path)
        if key in cache:
            hashes[i] = imagehash.hex_to_hash(cache[key])
            continue
        try:
            h = imagehash.phash(PILImage.open(path).convert("RGB"), hash_size=8)
        except Exception:
            h = imagehash.phash(PILImage.new("RGB", (8, 8)), hash_size=8)
        hashes[i] = h
        cache[key] = str(h)
        dirty = True

    if dirty:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache))
        except Exception:
            pass

    # ── 2. Build duplicate graph per class; union-find to get components ───────
    parent = list(range(len(samples)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    by_class: dict[int, list[int]] = collections.defaultdict(list)
    for idx, (_, label) in enumerate(samples):
        by_class[label].append(idx)

    for label, idxs in by_class.items():
        for ai in range(len(idxs)):
            for bi in range(ai + 1, len(idxs)):
                ia, ib = idxs[ai], idxs[bi]
                if (hashes[ia] - hashes[ib]) <= threshold:
                    union(ia, ib)

    # ── 3. Group indices by (class, component_root) ───────────────────────────
    comp_groups: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for idx, (_, label) in enumerate(samples):
        comp_groups[(label, find(idx))].append(idx)

    # ── 4. Assign each component to one split, stratified by class ─────────────
    comps_by_class: dict[int, list[list[int]]] = collections.defaultdict(list)
    for (label, _root), members in comp_groups.items():
        comps_by_class[label].append(members)

    rng = random.Random(seed)
    train_idx, val_idx, test_idx = [], [], []

    for label, comp_list in comps_by_class.items():
        rng.shuffle(comp_list)
        n = len(comp_list)
        n_test = max(1, int(round(n * test_split)))
        n_val  = max(1, int(round(n * val_split)))
        n_val  = min(n_val, max(0, n - n_test - 1))
        for members in comp_list[:n_test]:
            test_idx.extend(members)
        for members in comp_list[n_test:n_test + n_val]:
            val_idx.extend(members)
        for members in comp_list[n_test + n_val:]:
            train_idx.extend(members)

    return train_idx, val_idx, test_idx


def _seed_worker(worker_id):
    """Seed each DataLoader worker's RNG so augmentation is reproducible.

    torch sets a distinct base seed per worker (derived from the main process's
    seeded generator); we propagate it to numpy/random which some transforms use.
    """
    import random as _random

    import numpy as _np

    s = torch.initial_seed() % (2 ** 32)
    _np.random.seed(s)
    _random.seed(s)


def build_dataloaders(cfg):
    """Build train/val/test dataloaders and return them with the class names.

    Returns
    -------
    loaders : dict with keys "train", "val", "test"
    class_names : list[str]
    """
    raw_dir = cfg.raw_dir
    # Only count class sub-folders, so a lone .gitkeep (or stray files) doesn't
    # read as "has data" and then fail confusingly inside ImageFolder.
    class_dirs = [d for d in raw_dir.iterdir() if d.is_dir()] if raw_dir.exists() else []
    if not class_dirs:
        raise FileNotFoundError(
            f"No class sub-folders found in {raw_dir}. Add one sub-folder of "
            f"images per plant class first (see README / scripts/check_data.py)."
        )

    base = ImageFolder(str(raw_dir))  # no transform; subsets apply their own
    class_names = base.classes
    targets = [s[1] for s in base.samples]

    train_tf, eval_tf = build_transforms(cfg.data.image_size)
    deduplicate = getattr(cfg.data, "deduplicate", False)
    if deduplicate:
        train_idx, val_idx, test_idx = _component_stratified_split(
            base.samples, cfg.data.val_split, cfg.data.test_split, cfg.data.seed
        )
    else:
        train_idx, val_idx, test_idx = _stratified_split(
            targets, cfg.data.val_split, cfg.data.test_split, cfg.data.seed
        )

    datasets = {
        "train": _TransformedSubset(base, train_idx, train_tf),
        "val": _TransformedSubset(base, val_idx, eval_tf),
        "test": _TransformedSubset(base, test_idx, eval_tf),
    }

    g = torch.Generator()
    g.manual_seed(cfg.data.seed)
    loaders = {
        split: DataLoader(
            ds,
            batch_size=cfg.data.batch_size,
            shuffle=(split == "train"),
            num_workers=cfg.data.num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=g if split == "train" else None,
            worker_init_fn=_seed_worker,
        )
        for split, ds in datasets.items()
    }
    return loaders, class_names
