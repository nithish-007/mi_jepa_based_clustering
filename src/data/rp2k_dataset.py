"""
Base RP2K Dataset reading from meta_100.csv (curated 100-class subset).
Images come from the train/ directory only, split into train/val/test by the
prepare_dataset.py script.
"""
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class RP2KDataset(Dataset):
    """
    RP2K dataset filtered to the curated 100-class subset.

    Args:
        meta_csv: Path to meta_100.csv.
        split: 'train', 'val', or 'test'.
        transform: Torchvision transform to apply.
    """

    def __init__(
        self,
        meta_csv: str,
        split: str = "train",
        transform=None,
    ):
        self.transform = transform

        df = pd.read_csv(meta_csv)
        df = df[df["split"] == split].reset_index(drop=True)

        self.image_paths = df["image_path"].tolist()
        self.labels = df["class_name"].tolist()
        self.class_ids = df["class_id"].astype(int).tolist()

        # Build class_to_indices mapping (needed for bag construction)
        self.class_to_indices = defaultdict(list)
        for idx, cid in enumerate(self.class_ids):
            self.class_to_indices[cid].append(idx)

        self.num_classes = len(self.class_to_indices)
        self.all_class_ids = sorted(self.class_to_indices.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.class_ids[idx], idx

    def get_class_indices(self, class_id: int):
        """Return all dataset indices belonging to a given class."""
        return self.class_to_indices[class_id]
