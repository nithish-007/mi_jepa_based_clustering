from __future__ import annotations

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from src.data.datamodule import ProductDataModule
from src.training.callbacks import MLflowRunSetup, ValidationPlots
from src.training.checkpoints import FILENAME, checkpoint_dir, resolve_checkpoint
from src.training.module import ProductClusteringModule


def train(config: dict) -> dict:
    L.seed_everything(int(config["seed"]), workers=True)
    torch.set_float32_matmul_precision("high")

    ckpt_cfg, stop_cfg, mlflow_cfg, trainer_cfg = (config[k] for k in ("checkpoint", "early_stopping", "mlflow", "trainer"))
    directory = checkpoint_dir(config)
    resume = resolve_checkpoint(ckpt_cfg.get("resume", "auto"), directory)
    run_id = None
    if resume is not None:
        run_id = torch.load(resume, map_location="cpu", weights_only=False).get("mlflow_run_id")
        print(f"Resuming from {resume} (MLflow run {run_id})")

    module = ProductClusteringModule(config)
    datamodule = ProductDataModule(config, patch_size=module.net.backbone.patch_size)
    logger = MLFlowLogger(
        experiment_name=mlflow_cfg["experiment"],
        tracking_uri=mlflow_cfg["tracking_uri"],
        run_name=config["experiment"]["name"],
        run_id=run_id,
        tags={"stage": "train", "experiment_name": config["experiment"]["name"]},
    )
    checkpointing = ModelCheckpoint(
        dirpath=directory,
        filename=FILENAME,
        monitor=ckpt_cfg["monitor"],
        mode=ckpt_cfg["mode"],
        save_top_k=int(ckpt_cfg["save_top_k"]),
        save_last=True,
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )
    callbacks = [
        checkpointing,
        EarlyStopping(monitor=ckpt_cfg["monitor"], mode=ckpt_cfg["mode"], patience=int(stop_cfg["patience"]),
                      min_delta=float(stop_cfg.get("min_delta", 0.0))),
        LearningRateMonitor(logging_interval="step"),
        MLflowRunSetup(config, resumed_from=str(resume) if resume else None),
        ValidationPlots(int(mlflow_cfg.get("plot_every_n_epochs", 1))),
    ]
    trainer = L.Trainer(
        default_root_dir=str(directory),
        max_epochs=int(trainer_cfg["max_epochs"]),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=1,
        precision=trainer_cfg["precision"],
        accumulate_grad_batches=int(trainer_cfg.get("accumulate_grad_batches", 1)),
        gradient_clip_val=trainer_cfg.get("gradient_clip_val"),
        limit_train_batches=trainer_cfg.get("limit_train_batches") or 1.0,
        limit_val_batches=trainer_cfg.get("limit_val_batches") or 1.0,
        log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 50)),
        num_sanity_val_steps=0,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=str(resume) if resume else None)
    score = checkpointing.best_model_score
    return {
        "best_checkpoint": checkpointing.best_model_path,
        "best_score": float(score) if score is not None else None,
        "last_checkpoint": str(directory / "last.ckpt"),
        "mlflow_run_id": logger.run_id,
        "stopped_epoch": trainer.current_epoch,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
    }
