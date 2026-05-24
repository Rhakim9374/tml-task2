"""CIFAR-100 loaders and probe-set construction.

The defender knows the exact 40k indices used to train the target
(`target_model/train_main_idx.json`). The 10k remaining indices form a
holdout set that the target never saw — useful as the "test" half of
Dataset Inference (S2) and as an additional probe for other signals.
"""

import json
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms


MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


def eval_transform() -> transforms.Compose:
    """Deterministic normalize-only transform for scoring (no augmentation)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def train_aug_transform() -> transforms.Compose:
    """Target's exact training augmentation: hflip + biased random crop.

    bias_x=0.5, bias_y=-0.25, jitter=0.25 (per assignment PDF).
    Implemented as: reflect-pad=4, then a 32x32 crop whose top-left is
    sampled from a small window around the biased center.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        BiasedRandomCrop(size=32, pad=4, bias_x=0.5, bias_y=-0.25, jitter=0.25),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class BiasedRandomCrop:
    """Reflect-pad then crop a fixed-size patch centered around (bias_x, bias_y) with jitter.

    bias_x, bias_y are normalized offsets in [-1, 1] from the image center.
    jitter is the half-width of the uniform noise added to the crop top-left.
    """

    def __init__(self, size: int, pad: int, bias_x: float, bias_y: float, jitter: float):
        self.size = size
        self.pad = pad
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.jitter = jitter
        self.pad_t = transforms.Pad(pad, padding_mode="reflect")

    def __call__(self, img):
        img = self.pad_t(img)
        w, h = img.size  # PIL
        max_x = w - self.size
        max_y = h - self.size
        # center of valid crop range
        cx = max_x / 2 * (1 + self.bias_x)
        cy = max_y / 2 * (1 + self.bias_y)
        jx = (torch.rand(1).item() * 2 - 1) * self.jitter * max_x / 2
        jy = (torch.rand(1).item() * 2 - 1) * self.jitter * max_y / 2
        x = int(max(0, min(max_x, round(cx + jx))))
        y = int(max(0, min(max_y, round(cy + jy))))
        return img.crop((x, y, x + self.size, y + self.size))


def load_train_main_idx(target_dir: str = "target_model") -> list[int]:
    with open(Path(target_dir) / "train_main_idx.json") as f:
        return json.load(f)


def get_cifar100_splits(
    data_root: str = "./cifar100_data",
    target_dir: str = "target_model",
) -> Tuple[Subset, Subset, datasets.CIFAR100]:
    """Return (train_main, holdout, test) — all with eval transform, no aug.

    train_main: 40k indices the target trained on
    holdout: 10k remaining train-split indices the target never saw
    test: official 10k CIFAR-100 test set
    """
    tfm = eval_transform()
    full_train = datasets.CIFAR100(root=data_root, train=True, download=True, transform=tfm)
    test = datasets.CIFAR100(root=data_root, train=False, download=True, transform=tfm)

    train_idx = set(load_train_main_idx(target_dir))
    assert len(train_idx) == 40_000, f"Expected 40k train indices, got {len(train_idx)}"
    all_train = set(range(len(full_train)))
    holdout_idx = sorted(all_train - train_idx)
    assert len(holdout_idx) == 10_000, f"Expected 10k holdout, got {len(holdout_idx)}"

    train_main = Subset(full_train, sorted(train_idx))
    holdout = Subset(full_train, holdout_idx)
    return train_main, holdout, test


def get_augmented_train_probe(
    data_root: str = "./cifar100_data",
    target_dir: str = "target_model",
) -> Subset:
    """Same 40k indices as the target trained on, but with training-time augmentation
    applied. Used optionally for TTA in S1."""
    tfm = train_aug_transform()
    full_train = datasets.CIFAR100(root=data_root, train=True, download=True, transform=tfm)
    train_idx = sorted(load_train_main_idx(target_dir))
    return Subset(full_train, train_idx)


def get_ood_probe(
    data_root: str = "./cifar10_data",
    n_samples: int = 5000,
) -> Subset:
    """CIFAR-10 (resized none — already 32x32) as OOD probe. Used in S1 to catch
    distilled students that may have been extracted with OOD transfer data."""
    tfm = eval_transform()
    full = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tfm)
    # deterministic sample for reproducibility
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(full), generator=g)[:n_samples].tolist()
    return Subset(full, idx)


def make_loader(ds, batch_size: int = 512, num_workers: int = 4, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
