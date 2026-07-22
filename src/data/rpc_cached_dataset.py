"""
Cached RPC bag datasets — read pre-extracted bbox JPEGs from disk.

Mirrors the inter-bag / intra-bag / dual-bag pattern but consumes the manifest
produced by `scripts/preprocess_crops.py` instead of cropping HD source images
at every step.

Why: live cropping reads multi-megabyte JPEGs from the source dataset every
__getitem__, which limits training to ~0.13 it/s on this disk. Cached crops are
~20 KB each, so loading is ~13x faster (1.7 batches/s at batch=16, K=4, 8 workers).
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


def _load_manifest(
    manifest_path: str | Path,
    splits: tuple[str, ...] | None = None,
) -> tuple[
    dict[str, dict[int, list[tuple[int, str]]]],  # crops_by_image[split][image_id] -> [(cat, path)]
    dict[str, dict[int, list[str]]],              # crops_by_cat[split][cat] -> [path]
    dict[int, str],                               # categories (cat_id -> name placeholder)
]:
    """Read manifest CSV once and bucket rows by image and by category."""
    crops_by_image: dict[str, dict[int, list[tuple[int, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    crops_by_cat: dict[str, dict[int, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    cats: set[int] = set()

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            if splits is not None and split not in splits:
                continue
            image_id = int(row["image_id"])
            category_id = int(row["category_id"])
            crop_path = row["crop_path"]
            crops_by_image[split][image_id].append((category_id, crop_path))
            crops_by_cat[split][category_id].append(crop_path)
            cats.add(category_id)

    # Freeze defaultdicts -> plain dicts (smaller, safer)
    crops_by_image = {s: dict(v) for s, v in crops_by_image.items()}
    crops_by_cat = {s: dict(v) for s, v in crops_by_cat.items()}
    categories = {c: f"cat_{c}" for c in sorted(cats)}
    return crops_by_image, crops_by_cat, categories


class RPCCachedInterBagDataset(Dataset):
    """
    K-instance bag: K distinct categories from a single checkout image.

    Reads pre-extracted crops from disk via the manifest.

    Args:
        crops_by_image: split -> image_id -> list[(category_id, crop_path)].
        split: 'val' or 'test' (checkout images live there).
        bag_size: K.
        transform: torchvision transform.
    """

    def __init__(
        self,
        crops_by_image: dict[str, dict[int, list[tuple[int, str]]]],
        split: str,
        bag_size: int,
        transform,
    ):
        if split not in crops_by_image:
            raise ValueError(f"Split {split!r} not present in manifest.")
        if bag_size < 2:
            raise ValueError(f"bag_size must be >=2, got {bag_size}")
        self.bag_size = bag_size
        self.transform = transform

        # Per-image: cat_id -> list[crop_path]
        self._by_image: dict[int, dict[int, list[str]]] = {}
        for img_id, rows in crops_by_image[split].items():
            by_cat: dict[int, list[str]] = defaultdict(list)
            for cat, path in rows:
                by_cat[cat].append(path)
            if len(by_cat) >= bag_size:
                self._by_image[img_id] = dict(by_cat)

        self._image_ids: list[int] = list(self._by_image.keys())
        if not self._image_ids:
            raise RuntimeError(
                f"No images in split={split!r} have >= {bag_size} distinct categories."
            )

    def __len__(self) -> int:
        return len(self._image_ids)

    def __getitem__(self, idx):
        img_id = self._image_ids[idx]
        by_cat = self._by_image[img_id]
        cats = random.sample(list(by_cat.keys()), self.bag_size)

        instances: list[torch.Tensor] = []
        labels: list[int] = []
        for c in cats:
            path = random.choice(by_cat[c])
            crop = Image.open(path).convert("RGB")
            instances.append(self.transform(crop))
            labels.append(int(c))
        return torch.stack(instances), torch.tensor(labels, dtype=torch.long)


class RPCCachedIntraBagDataset(Dataset):
    """
    K-instance bag: K crops from the SAME product category.

    Args:
        crops_by_cat: split -> category_id -> list[crop_path].
        split: 'train' is the cleanest pool (single-product studio crops).
        bag_size: K.
        transform: torchvision transform.
        length: virtual epoch length.
    """

    def __init__(
        self,
        crops_by_cat: dict[str, dict[int, list[str]]],
        split: str,
        bag_size: int,
        transform,
        length: int,
    ):
        if split not in crops_by_cat:
            raise ValueError(f"Split {split!r} not present in manifest.")
        self.bag_size = bag_size
        self.transform = transform
        self.length = length
        self._by_cat: dict[int, list[str]] = {
            c: paths for c, paths in crops_by_cat[split].items() if len(paths) >= 1
        }
        self._cats: list[int] = list(self._by_cat.keys())
        if not self._cats:
            raise RuntimeError(f"No categories with crops in split={split!r}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        cat = random.choice(self._cats)
        paths = self._by_cat[cat]
        if len(paths) >= self.bag_size:
            chosen = random.sample(paths, self.bag_size)
        else:
            chosen = random.choices(paths, k=self.bag_size)
        instances = [self.transform(Image.open(p).convert("RGB")) for p in chosen]
        return (
            torch.stack(instances),
            torch.full((self.bag_size,), cat, dtype=torch.long),
        )


class RPCCachedDualBagDataset(Dataset):
    """
    Pairs a cached inter-bag with a cached intra-bag per sample, matching the
    existing `DualBagDataset` 4-tuple API:
        (inter_imgs, inter_labels, intra_imgs, intra_labels)

    Args:
        manifest_path: dataset/rpc_crops_manifest.csv produced by preprocess_crops.py.
        bag_size: K.
        transform: torchvision transform applied to every crop.
        inter_split: 'val' or 'test'. Default 'test' (24K checkout images, large).
        intra_split: 'train' (single-product studio crops; ~270 per SKU).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        bag_size: int,
        transform,
        inter_split: str = "test",
        intra_split: str = "train",
    ):
        crops_by_image, crops_by_cat, categories = _load_manifest(
            manifest_path, splits=(inter_split, intra_split)
        )
        self.inter = RPCCachedInterBagDataset(
            crops_by_image, inter_split, bag_size, transform
        )
        self.intra = RPCCachedIntraBagDataset(
            crops_by_cat, intra_split, bag_size, transform, length=len(self.inter)
        )
        self.num_categories = len(categories)
        # Keep references in case caller wants to introspect
        self._crops_by_image = crops_by_image
        self._crops_by_cat = crops_by_cat
        self._categories = categories

    def __len__(self) -> int:
        return len(self.inter)

    def __getitem__(self, idx):
        inter_imgs, inter_labels = self.inter[idx]
        intra_imgs, intra_labels = self.intra[idx]
        return inter_imgs, inter_labels, intra_imgs, intra_labels


class RPCCachedCropEvalDataset(Dataset):
    """
    Flat (crop, label) eval dataset over cached crops.

    Used by `evaluate_clustering_rpc.py` to score the encoder by sampling a
    balanced subset of bbox crops per category from a chosen split.

    Args:
        manifest_path: dataset/rpc_crops_manifest.csv.
        split: which split to evaluate on (default 'test').
        transform: torchvision eval transform.
        samples_per_class: cap per-category to this many crops (None = use all).
        seed: RNG seed for reproducible sampling.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        transform,
        samples_per_class: int | None = None,
        seed: int = 0,
    ):
        _, crops_by_cat, _ = _load_manifest(manifest_path, splits=(split,))
        if split not in crops_by_cat:
            raise ValueError(f"Split {split!r} not present in manifest.")

        rng = random.Random(seed)
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
        for cat, paths in sorted(crops_by_cat[split].items()):
            pool = list(paths)
            if samples_per_class is not None and len(pool) > samples_per_class:
                pool = rng.sample(pool, samples_per_class)
            for p in pool:
                self.samples.append((p, int(cat)))
        rng.shuffle(self.samples)
        self.num_categories = len(crops_by_cat[split])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def dual_bag_collate_fn(batch):
    """
    Collate for `RPCCachedDualBagDataset`.

    Returns:
        inter_imgs:    (B, K, C, H, W)
        inter_labels:  (B, K)
        intra_imgs:    (B, K, C, H, W)
        intra_labels:  (B, K)
    """
    inter_imgs = torch.stack([b[0] for b in batch])
    inter_labels = torch.stack([b[1] for b in batch])
    intra_imgs = torch.stack([b[2] for b in batch])
    intra_labels = torch.stack([b[3] for b in batch])
    return inter_imgs, inter_labels, intra_imgs, intra_labels
