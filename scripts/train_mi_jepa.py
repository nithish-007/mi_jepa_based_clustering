"""
Train MI-JEPA on RPC checkout images as native multi-instance bags.

Reads pre-extracted bbox crops via `dataset/rpc_crops_manifest.csv` (see
`scripts/preprocess_crops.py`). Config-driven via `configs/training.yaml`.
"""
from pathlib import Path
import sys

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.rpc_cached_dataset import RPCCachedDualBagDataset, dual_bag_collate_fn
from src.data.transforms import get_ssl_transform
from src.models.mi_jepa import MIJEPAModule


def load_config(path: str = "configs/training.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--fast_dev", action="store_true",
                        help="Run a 5-step pl.Trainer fast-dev for smoke testing.")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Override training.max_epochs for short runs.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
    es_cfg = cfg["early_stopping"]
    ckpt_cfg = cfg["checkpoint"]
    mlflow_cfg = cfg["mlflow"]

    if args.max_epochs is not None:
        train_cfg["max_epochs"] = args.max_epochs

    pl.seed_everything(train_cfg["seed"])

    # --- Datasets (cached crops) ---
    manifest = data_cfg["manifest_csv"]
    if not Path(manifest).exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest!r}. Run "
            "`python scripts/preprocess_crops.py` first."
        )
    transform = get_ssl_transform(data_cfg["img_size"])
    train_dataset = RPCCachedDualBagDataset(
        manifest,
        bag_size=data_cfg["bag_size"],
        transform=transform,
        inter_split=data_cfg["inter_split"],
        intra_split=data_cfg["intra_split"],
    )
    val_dataset = RPCCachedDualBagDataset(
        manifest,
        bag_size=data_cfg["bag_size"],
        transform=transform,
        inter_split=data_cfg["val_inter_split"],
        intra_split=data_cfg["intra_split"],
    )
    print(f"RPC: {train_dataset.num_categories} categories")
    print(f"Train bags (inter src): {len(train_dataset)}")
    print(f"Val   bags (inter src): {len(val_dataset)}")

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

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_cfg["dirpath"],
            filename="mi-jepa-rpc-{epoch:03d}-{val/loss:.4f}",
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

    logger = MLFlowLogger(
        experiment_name=mlflow_cfg["experiment_name"],
        run_name=mlflow_cfg["run_name"],
        tracking_uri=mlflow_cfg["tracking_uri"],
        log_model=False,
    )

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
        fast_dev_run=5 if args.fast_dev else False,
    )

    trainer.fit(model, train_loader, val_loader)
    print(f"Training complete. Best checkpoint: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
