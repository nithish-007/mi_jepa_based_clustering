"""A tiny synthetic RP2K-style tree and a tiny model config, so tests run on CPU in seconds."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import PROJECT_ROOT, load_config


def make_image_tree(root: Path, n_skus: int = 24, images_per_sku: int = 6, seed: int = 0) -> Path:
    """``root/all/<train|test>/<sku>/<n>.jpg``. Each SKU has its own colour and stripe pattern."""
    rng = np.random.default_rng(seed)
    for s in range(n_skus):
        colour = rng.integers(0, 255, 3)
        stripe = int(rng.integers(2, 12))
        sku = f"brand{s % 6}_{100 + 50 * (s // 6)}ml"
        for i in range(images_per_sku):
            official = "train" if i % 2 == 0 else "test"
            folder = root / "all" / official / sku
            folder.mkdir(parents=True, exist_ok=True)
            pixels = np.tile(colour, (48, 40, 1)).astype(np.float64)
            pixels[::stripe] = 255 - pixels[::stripe]
            pixels += rng.normal(0, 12, pixels.shape)
            Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).save(folder / f"{i}.jpg", quality=95)
    others = root / "all" / "train" / "others"
    others.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)).save(others / "x.jpg")
    return root


def data_config(tmp: Path) -> dict:
    config = load_config("configs/data.yaml")
    config.update({"raw_root": str(tmp / "raw"), "output_dir": str(tmp / "splits"), "split_fractions": {"train": 0.5, "val": 0.25}})
    return config


def train_config(tmp: Path) -> dict:
    config = copy.deepcopy(load_config("configs/train.yaml"))
    config["experiment"]["name"] = "tiny"
    config["data"].update({"root": str(tmp / "raw"), "manifest": str(tmp / "splits" / "manifest.csv"),
                           "image_size": 64, "num_workers": 0})
    config["data"]["sampler"].update({"products_per_batch": 4, "images_per_product": 2, "batches_per_epoch": 3})
    config["data"]["validation"].update({"batch_size": 16, "max_images_per_sku": None, "num_workers": 0})
    config["model"]["backbone"] = {
        "name": "vit_tiny_patch16_224", "checkpoint": None, "pretrained": False, "trainable_blocks": 2,
        "pooling": "cls", "grad_checkpointing": False,
        "kwargs": {"patch_size": 8, "embed_dim": 64, "depth": 4, "num_heads": 2},
    }
    config["model"]["ijepa"]["mask"]["min_keep"] = 2
    config["model"]["ijepa"]["predictor"].update({"embed_dim": 32, "depth": 1, "num_heads": 2})
    for head in ("invariance_head", "discrimination_head"):
        config["model"][head].update({"hidden_dims": [32], "out_dim": 16})
    config["model"]["discrimination_head"]["energy_hidden_dims"] = [16]
    config["trainer"].update({"max_epochs": 2, "precision": "32-true", "accelerator": "cpu", "log_every_n_steps": 1})
    config["checkpoint"]["dir"] = str(tmp / "checkpoints")
    config["mlflow"].update({"tracking_uri": f"sqlite:///{(tmp / 'mlflow.db').as_posix()}", "log_grad_norm_every_n_steps": 1})
    config["_source"] = str(PROJECT_ROOT / "configs" / "train.yaml")
    return config
