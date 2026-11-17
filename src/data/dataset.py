from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ProductImageDataset(Dataset):
    """Images with integer labels. ``label`` is -1 when the identity is unknown."""

    def __init__(self, paths: list[Path], labels: np.ndarray | None, transform):
        self.paths = [Path(p) for p in paths]
        self.labels = np.full(len(self.paths), -1, dtype=np.int64) if labels is None else np.asarray(labels, dtype=np.int64)
        self.transform = transform

    @classmethod
    def from_manifest(cls, frame: pd.DataFrame, root: Path, transform) -> "ProductImageDataset":
        labels, _ = encode_labels(frame["sku_id"])
        return cls([root / p for p in frame["image_path"]], labels, transform)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        with Image.open(self.paths[index]) as handle:
            image = handle.convert("RGB")
        return {"image": self.transform(image), "label": torch.tensor(self.labels[index]), "index": torch.tensor(index)}


def encode_labels(values: pd.Series) -> tuple[np.ndarray, list[str]]:
    codes, names = pd.factorize(values, sort=True)
    return codes.astype(np.int64), [str(n) for n in names]
