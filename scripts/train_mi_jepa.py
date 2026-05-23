"""
Train MI-JEPA (dual-stream) on the curated RP2K-100 dataset.
Config-driven via configs/training.yaml.
"""
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.bag_dataset import DualBagDataset, dual_bag_collate_fn
from src.data.rp2k_dataset import RP2KDataset
from src.data.transforms import get_ssl_transform
from src.models.mi_jepa import MIJEPAModule


def load_config(path: str = "configs/training.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
    es_cfg = cfg["early_stopping"]
    ckpt_cfg = cfg["checkpoint"]
    mlflow_cfg = cfg["mlflow"]

    pl.seed_everything(train_cfg["seed"])

    # --- Datasets ---
    transform = get_ssl_transform(data_cfg["img_size"])

    train_base = RP2KDataset(meta_csv=data_cfg["meta_csv"], split="train", transform=transform)
    val_base = RP2KDataset(meta_csv=data_cfg["meta_csv"], split="val", transform=transform)

    print(f"Train: {len(train_base)} images, {train_base.num_classes} classes")
    print(f"Val  : {len(val_base)} images, {val_base.num_classes} classes")

    train_dataset = DualBagDataset(train_base, bag_size=data_cfg["bag_size"])
    val_dataset = DualBagDataset(val_base, bag_size=data_cfg["bag_size"])

    loader_kwargs = dict(
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        collate_fn=dual_bag_collate_fn,
        persistent_workers=data_cfg["num_workers"] > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    # --- Model ---
    model = MIJEPAModule(
        embed_dim=model_cfg["embed_dim"],
        model_name=model_cfg["name"],
        pretrained=model_cfg["pretrained"],
        bag_size=data_cfg["bag_size"],
        mask_ratio=model_cfg["mask_ratio"],
        cia_layers=model_cfg["cia_layers"],
        cia_heads=model_cfg["cia_heads"],
        pred_layers=model_cfg["pred_layers"],
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        warmup_epochs=train_cfg["warmup_epochs"],
        max_epochs=train_cfg["max_epochs"],
        base_momentum=train_cfg["base_momentum"],
        lambda_jepa=loss_cfg["lambda_jepa"],
        lambda_energy=loss_cfg["lambda_energy"],
        lambda_reg=loss_cfg["lambda_reg"],
        energy_margin_low=loss_cfg["energy_margin_low"],
        energy_margin_high=loss_cfg["energy_margin_high"],
    )

    # --- Callbacks ---
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_cfg["dirpath"],
            filename="mi-jepa-{epoch:03d}-{val/loss:.4f}",
            save_top_k=ckpt_cfg["save_top_k"],
            monitor=ckpt_cfg["monitor"],
            mode=ckpt_cfg["mode"],
            every_n_epochs=ckpt_cfg["save_every"],
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    if es_cfg["enabled"]:
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg["monitor"],
                patience=es_cfg["patience"],
                mode=es_cfg["mode"],
                verbose=True,
            )
        )

    # --- MLflow Logger ---
    logger = MLFlowLogger(
        experiment_name=mlflow_cfg["experiment_name"],
        run_name=mlflow_cfg["run_name"],
        tracking_uri=mlflow_cfg["tracking_uri"],
        log_model=False,
    )

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        accelerator="auto",
        devices=1,
        precision="16-mixed" if train_cfg["fp16"] else "32-true",
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=train_cfg["gradient_clip_val"],
        accumulate_grad_batches=train_cfg["accum_grad"],
        deterministic=False,
    )

    trainer.fit(model, train_loader, val_loader)
    print(f"Training complete. Best checkpoint: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
