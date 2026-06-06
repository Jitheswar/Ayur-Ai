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


def build_dataloaders(cfg):
    """Build train/val/test dataloaders and return them with the class names.

    Returns
    -------
    loaders : dict with keys "train", "val", "test"
    class_names : list[str]
    """
    raw_dir = cfg.raw_dir
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        raise FileNotFoundError(
            f"No images found in {raw_dir}. Add one sub-folder of images per "
            f"plant class first (see README / scripts/check_data.py)."
        )

    base = ImageFolder(str(raw_dir))  # no transform; subsets apply their own
    class_names = base.classes
    targets = [s[1] for s in base.samples]

    train_tf, eval_tf = build_transforms(cfg.data.image_size)
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
        )
        for split, ds in datasets.items()
    }
    return loaders, class_names
