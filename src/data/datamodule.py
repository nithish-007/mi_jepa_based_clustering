from __future__ import annotations

import lightning as L
from lightly.data.collate import IJEPAMaskCollator
from torch.utils.data import DataLoader

from src.config import resolve_path
from src.data.dataset import ProductImageDataset, encode_labels
from src.data.sampler import PKBatchSampler
from src.data.splits import load_manifest, select_records
from src.data.transforms import eval_transform, train_transform


def build_mask_collator(data_cfg: dict, mask_cfg: dict, patch_size: int) -> IJEPAMaskCollator:
    """I-JEPA multi-block masks (Lightly port of the official sampler), drawn per batch."""
    return IJEPAMaskCollator(
        input_size=(int(data_cfg["image_size"]),) * 2,
        patch_size=int(patch_size),
        pred_mask_scale=tuple(mask_cfg["target_scale"]),
        aspect_ratio=tuple(mask_cfg["target_aspect_ratio"]),
        npred=int(mask_cfg["num_targets"]),
        enc_mask_scale=tuple(mask_cfg["context_scale"]),
        nenc=1,
        min_keep=int(mask_cfg["min_keep"]),
        allow_overlap=False,
    )


def loader_kwargs(num_workers: int) -> dict:
    return {"num_workers": num_workers, "pin_memory": True, "persistent_workers": num_workers > 0}


class ProductDataModule(L.LightningDataModule):
    """Train batches are P x K product batches plus I-JEPA masks. Validation is a capped val split."""

    def __init__(self, config: dict, patch_size: int):
        super().__init__()
        self.cfg = config
        self.data_cfg = config["data"]
        self.patch_size = int(patch_size)
        self.root = resolve_path(self.data_cfg["root"])

    def setup(self, stage: str | None = None) -> None:
        manifest = load_manifest(self.data_cfg["manifest"])
        train = select_records(manifest, "train", self.data_cfg.get("max_train_skus"))
        val_cfg = self.data_cfg["validation"]
        val = select_records(manifest, "val", val_cfg.get("max_skus"), val_cfg.get("max_images_per_sku"))
        self.train_set = ProductImageDataset.from_manifest(train, self.root, train_transform(self.data_cfg))
        self.val_set = ProductImageDataset.from_manifest(val, self.root, eval_transform(self.data_cfg))
        self.train_families, _ = encode_labels(train["family"])
        self.val_frame = val

    def train_dataloader(self) -> DataLoader:
        sampler_cfg = self.data_cfg["sampler"]
        batch_sampler = PKBatchSampler(
            self.train_set.labels,
            self.train_families,
            products_per_batch=sampler_cfg["products_per_batch"],
            images_per_product=sampler_cfg["images_per_product"],
            batches_per_epoch=sampler_cfg.get("batches_per_epoch"),
            same_family_prob=sampler_cfg.get("same_family_prob", 0.0),
            seed=int(self.cfg["seed"]),
        )
        collate = build_mask_collator(self.data_cfg, self.cfg["model"]["ijepa"]["mask"], self.patch_size)
        return DataLoader(self.train_set, batch_sampler=batch_sampler, collate_fn=collate,
                          **loader_kwargs(int(self.data_cfg["num_workers"])))

    def val_dataloader(self) -> DataLoader:
        val_cfg = self.data_cfg["validation"]
        return DataLoader(self.val_set, batch_size=int(val_cfg["batch_size"]), shuffle=False,
                          **loader_kwargs(int(val_cfg.get("num_workers", 0))))
