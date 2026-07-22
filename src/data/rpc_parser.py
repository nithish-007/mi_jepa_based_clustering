"""
RPC (Retail Product Checkout) dataset parser.

RPC structure:
- `train2019/`: ~53K single-product studio images (1 product per image, 4 cameras,
  multiple angles per SKU). Used as the intra-bag source.
- `val2019/`: ~6K checkout images (multiple products per image, COCO bbox annotations).
  Used as inter-bag source for validation.
- `test2019/`: ~24K checkout images (multiple products per image). Used as inter-bag
  source for training (larger pool) and held-out eval.

COCO bbox format: [x, y, width, height] in absolute pixel coordinates.

This module loads the three `instances_*.json` COCO files and builds compact
in-memory indices suitable for fast random sampling during training:

    RPCIndex
    ├── images: {image_id -> {file_name, width, height, split}}
    ├── ann_by_image: {image_id -> list[Annotation]}            # for inter-bag
    ├── images_by_category: {category_id -> list[image_id]}     # for intra-bag (train2019)
    ├── annotations_by_category: {category_id -> list[Annotation]}  # for intra-bag from crops
    └── categories: {category_id -> {name, supercategory}}

The full index is cached to `dataset/rpc_index.pkl` (pickled dataclass) to avoid
re-parsing the JSONs every run.
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class Annotation:
    """One bbox annotation."""
    ann_id: int                               # COCO `id` (unique within JSON)
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]  # x, y, w, h (COCO)
    split: str                                # 'train', 'val', 'test'


@dataclass
class RPCIndex:
    """In-memory index of the full RPC dataset."""
    # split -> {image_id -> {file_name, width, height}}
    images: dict[str, dict[int, dict]] = field(default_factory=dict)
    # split -> {image_id -> list[Annotation]}
    ann_by_image: dict[str, dict[int, list[Annotation]]] = field(default_factory=dict)
    # split -> {category_id -> list[image_id]}
    images_by_category: dict[str, dict[int, list[int]]] = field(default_factory=dict)
    # split -> {category_id -> list[Annotation]}  (useful if we ever crop from checkout images by class)
    annotations_by_category: dict[str, dict[int, list[Annotation]]] = field(default_factory=dict)
    # {category_id -> {'name': str, 'supercategory': str}}
    categories: dict[int, dict] = field(default_factory=dict)
    # Absolute path to the dataset root (e.g. ".../retail_product_checkout")
    root: str = ""

    @property
    def num_categories(self) -> int:
        return len(self.categories)

    def image_path(self, split: str, image_id: int) -> Path:
        """Absolute file path for an image."""
        meta = self.images[split][image_id]
        return Path(self.root) / f"{split}2019" / meta["file_name"]

    def categories_in_image(self, split: str, image_id: int) -> set[int]:
        """Distinct category ids annotated in a checkout image."""
        return {a.category_id for a in self.ann_by_image[split][image_id]}

    def list_image_ids(self, split: str) -> list[int]:
        return list(self.images[split].keys())


def _parse_split(json_path: Path, split: str) -> tuple[
    dict[int, dict],
    dict[int, list[Annotation]],
    dict[int, list[int]],
    dict[int, list[Annotation]],
    dict[int, dict],
]:
    """Parse one COCO instances_*.json file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = {}
    for img in data["images"]:
        images[img["id"]] = {
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
        }

    ann_by_image: dict[int, list[Annotation]] = defaultdict(list)
    images_by_category: dict[int, set[int]] = defaultdict(set)
    annotations_by_category: dict[int, list[Annotation]] = defaultdict(list)
    for a in data["annotations"]:
        x, y, w, h = a["bbox"]
        ann = Annotation(
            ann_id=int(a["id"]),
            image_id=int(a["image_id"]),
            category_id=int(a["category_id"]),
            bbox=(float(x), float(y), float(w), float(h)),
            split=split,
        )
        ann_by_image[ann.image_id].append(ann)
        images_by_category[ann.category_id].add(ann.image_id)
        annotations_by_category[ann.category_id].append(ann)

    categories = {
        int(c["id"]): {"name": c["name"], "supercategory": c.get("supercategory", "")}
        for c in data.get("categories", [])
    }

    # Convert sets to lists for stable random access
    images_by_category_list = {k: sorted(v) for k, v in images_by_category.items()}

    return (
        dict(images),
        dict(ann_by_image),
        images_by_category_list,
        dict(annotations_by_category),
        categories,
    )


def build_index(
    rpc_root: str | Path,
    splits: Iterable[str] = ("train", "val", "test"),
) -> RPCIndex:
    """Parse all COCO JSONs and build an `RPCIndex`."""
    rpc_root = Path(rpc_root)
    idx = RPCIndex(root=str(rpc_root.resolve()))

    canonical_categories: dict[int, dict] = {}

    for split in splits:
        json_path = rpc_root / f"instances_{split}2019.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Missing annotations file: {json_path}")
        images, ann_by_image, images_by_cat, ann_by_cat, cats = _parse_split(
            json_path, split
        )
        idx.images[split] = images
        idx.ann_by_image[split] = ann_by_image
        idx.images_by_category[split] = images_by_cat
        idx.annotations_by_category[split] = ann_by_cat
        if not canonical_categories:
            canonical_categories = cats

    idx.categories = canonical_categories
    return idx


def save_index(index: RPCIndex, path: str | Path) -> None:
    """Pickle the index to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(path: str | Path) -> RPCIndex:
    """Load a pickled index from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def get_or_build_index(
    rpc_root: str | Path,
    cache_path: str | Path = "dataset/rpc_index.pkl",
    force_rebuild: bool = False,
) -> RPCIndex:
    """Load cached index, or build & cache it from `rpc_root`."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_rebuild:
        return load_index(cache_path)
    idx = build_index(rpc_root)
    save_index(idx, cache_path)
    return idx
