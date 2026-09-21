"""
Train MI-JEPA on RPC checkout images as native multi-instance bags.

Reads pre-extracted bbox crops via `dataset/rpc_crops_manifest.csv` (see
`scripts/preprocess_crops.py`). Config-driven via `configs/training.yaml`.
"""
from pathlib import Path
import sys

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.plugins.io import TorchCheckpointIO
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.rpc_cached_dataset import RPCCachedDualBagDataset, dual_bag_collate_fn
from src.data.transforms import get_ssl_transform
from src.models.mi_jepa import MIJEPAModule


class DirectTorchCheckpointIO(TorchCheckpointIO):
    """Write checkpoints straight to disk instead of via an in-memory buffer.

    Lightning's default `_atomic_save` serialises into `io.BytesIO` first, which
    trips torch's zip-writer alignment check on a checkpoint this size
    ("RuntimeError: unexpected pos N vs M"). Saving to the path directly uses
    torch's file writer and also avoids a full extra copy in RAM.
    """

    def save_checkpoint(self, checkpoint, path, storage_options=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)


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

    # Top-K rolling checkpoints (+ last.ckpt), gated by save_every epochs.
    # NOTE: auto_insert_metric_name=False + no '/' in the template. The monitored
    # metric name is "val/loss"; if the '/' reached the filename, Lightning would
    # create a subfolder per checkpoint and leave empty folders behind on rotation.
    topk_ckpt = ModelCheckpoint(
        dirpath=ckpt_cfg["dirpath"],
        filename="mi-jepa-rpc-epoch{epoch:03d}-loss{val/loss:.4f}",
        auto_insert_metric_name=False,
        save_top_k=ckpt_cfg["save_top_k"],
        monitor=ckpt_cfg["monitor"],
        mode=ckpt_cfg["mode"],
        every_n_epochs=ckpt_cfg["save_every"],
        save_last=True,
    )
    # Single always-current best checkpoint, e.g. `best_loss_0.1234.ckpt`.
    # save_top_k=1 makes Lightning delete the previous best and write the new
    # one whenever val/loss improves, so the best model is unambiguous. Checked
    # every validation epoch (no every_n_epochs) so it's the true global best.
    best_ckpt = ModelCheckpoint(
        dirpath=ckpt_cfg["dirpath"],
        filename="best_loss_{val/loss:.4f}",
        auto_insert_metric_name=False,
        save_top_k=1,
        monitor=ckpt_cfg["monitor"],
        mode=ckpt_cfg["mode"],
        save_last=False,
    )
    callbacks = [
        topk_ckpt,
        best_ckpt,
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
        plugins=[DirectTorchCheckpointIO()],
        log_every_n_steps=10,
        gradient_clip_val=train_cfg["gradient_clip_val"],
        accumulate_grad_batches=train_cfg["accum_grad"],
        deterministic=False,
        fast_dev_run=5 if args.fast_dev else False,
    )

    trainer.fit(model, train_loader, val_loader)
    print("Training complete.")
    print(f"  Best checkpoint: {best_ckpt.best_model_path}")
    print(f"  Best {ckpt_cfg['monitor']}: {best_ckpt.best_model_score}")
    print(f"  Top-{ckpt_cfg['save_top_k']} + last saved in: {ckpt_cfg['dirpath']}")


if __name__ == "__main__":
    main()
