"""
Dual-stream bag datasets for MI-JEPA.

- InterBagDataset: bags of K different products (learn discriminative features).
- IntraBagDataset: bags of K views of the same product (learn invariant features).
- DualBagDataset: pairs an inter-bag and intra-bag per sample.
"""
import random

import torch
from torch.utils.data import Dataset

from src.data.rp2k_dataset import RP2KDataset


class InterBagDataset(Dataset):
    """
    Each sample is a bag of K images from K different product classes.
    Forces the model to learn what distinguishes each product.
    """

    def __init__(self, base_dataset: RP2KDataset, bag_size: int = 4):
        self.base = base_dataset
        self.bag_size = bag_size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        K = self.bag_size
        # Sample K distinct classes
        selected_classes = random.sample(self.base.all_class_ids, K)

        images = []
        labels = []
        for cid in selected_classes:
            indices = self.base.get_class_indices(cid)
            chosen_idx = random.choice(indices)
            img, label, _ = self.base[chosen_idx]
            images.append(img)
            labels.append(label)

        # (K, C, H, W), (K,)
        return torch.stack(images), torch.tensor(labels, dtype=torch.long)


class IntraBagDataset(Dataset):
    """
    Each sample is a bag of K images/views of the SAME product class.
    Forces the model to learn what's invariant within a product.

    If a class has fewer than K images, duplicates are augmented differently
    (relying on the stochastic SSL transform applied in the base dataset).
    """

    def __init__(self, base_dataset: RP2KDataset, bag_size: int = 4):
        self.base = base_dataset
        self.bag_size = bag_size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        K = self.bag_size
        # Pick a random class
        cid = random.choice(self.base.all_class_ids)
        indices = self.base.get_class_indices(cid)

        if len(indices) >= K:
            chosen = random.sample(indices, K)
        else:
            # Sample with replacement — different augmentations make them distinct
            chosen = random.choices(indices, k=K)

        images = []
        labels = []
        for i in chosen:
            img, label, _ = self.base[i]
            images.append(img)
            labels.append(label)

        # (K, C, H, W), (K,)  — all labels identical for intra-bag
        return torch.stack(images), torch.tensor(labels, dtype=torch.long)


class DualBagDataset(Dataset):
    """
    Wraps InterBagDataset and IntraBagDataset to return paired bags.
    Each sample returns (inter_bag, inter_labels, intra_bag, intra_labels).
    """

    def __init__(self, base_dataset: RP2KDataset, bag_size: int = 4):
        self.inter = InterBagDataset(base_dataset, bag_size)
        self.intra = IntraBagDataset(base_dataset, bag_size)

    def __len__(self):
        return len(self.inter)

    def __getitem__(self, idx):
        inter_imgs, inter_labels = self.inter[idx]
        intra_imgs, intra_labels = self.intra[idx]
        return inter_imgs, inter_labels, intra_imgs, intra_labels


def dual_bag_collate_fn(batch):
    """
    Collate for DualBagDataset.
    Returns:
        inter_imgs: (B, K, C, H, W)
        inter_labels: (B, K)
        intra_imgs: (B, K, C, H, W)
        intra_labels: (B, K)
    """
    inter_imgs = torch.stack([b[0] for b in batch])
    inter_labels = torch.stack([b[1] for b in batch])
    intra_imgs = torch.stack([b[2] for b in batch])
    intra_labels = torch.stack([b[3] for b in batch])
    return inter_imgs, inter_labels, intra_imgs, intra_labels
