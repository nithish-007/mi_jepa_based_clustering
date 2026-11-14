from __future__ import annotations

import math

import lightning as L
import torch

from src.config import fingerprint
from src.evaluation.metrics import recall_at_k
from src.inference.scoring import PairScorer, fit_calibration
from src.losses.discrimination import discrimination_loss
from src.losses.invariance import supervised_contrastive_loss
from src.models.network import ProductClusteringNet


def model_fingerprint(config: dict) -> str:
    return fingerprint({"model": config["model"], "image_size": config["data"]["image_size"]})


class ProductClusteringModule(L.LightningModule):
    """Joint training of the I-JEPA branch, the invariance head and the discrimination head.

    Every validation epoch clusters the validation split exactly as evaluation does
    (kNN candidates, calibrated pair probability, union-find) and logs B-cubed F1,
    which is what early stopping and checkpoint selection monitor.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config
        self.loss_cfg = config["loss"]
        self.net = ProductClusteringNet(config["model"], config["data"]["image_size"])
        self.calibration: dict | None = None
        self.validation_diagnostics: dict | None = None
        self._val_batches: list[dict] = []

    # ----------------------------------------------------------------- training
    def _weight(self, name: str) -> float:
        part = self.loss_cfg[name]
        if self.current_epoch < int(part.get("start_epoch", 0)):
            return 0.0
        return float(part["weight"])

    def training_step(self, batch, batch_idx):
        data, masks_context, masks_target = batch
        images, labels = data["image"], data["label"]
        out = self.net(images)
        logs, total = {}, images.new_zeros(())

        weight = self._weight("invariance")
        if weight:
            loss = supervised_contrastive_loss(out["z_inv"], labels, float(self.loss_cfg["invariance"]["temperature"]))
            total = total + weight * loss
            logs["train/invariance_loss"] = loss

        weight = self._weight("discrimination")
        if weight:
            cfg = self.loss_cfg["discrimination"]
            energy = self.net.discrimination_head.energy_matrix(out["z_dis"])
            parts = discrimination_loss(energy, labels, float(cfg["margin"]), float(cfg["ranking_weight"]), float(cfg["bce_weight"]))
            total = total + weight * parts["loss"]
            logs.update({f"train/discrimination_{k}": v for k, v in parts.items()})

        weight = self._weight("ijepa")
        if weight and self.net.ijepa is not None:
            parts = self.net.ijepa(self.net.backbone, images, masks_context, masks_target)
            total = total + weight * parts["loss"]
            logs.update({f"train/ijepa_{k}": v for k, v in parts.items()})

        if not logs:
            raise RuntimeError(f"No loss is active at epoch {self.current_epoch}. Check loss.*.weight and start_epoch.")
        logs["train/loss"] = total
        self.log_dict(logs, on_step=True, on_epoch=True, batch_size=len(labels))
        return total

    def ema_momentum(self) -> float:
        ema = self.cfg["model"]["ijepa"]["ema"]
        total = max(1, int(self.trainer.estimated_stepping_batches))
        progress = min(1.0, self.global_step / total)
        return float(ema["start"]) + (float(ema["end"]) - float(ema["start"])) * progress

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        if self.net.ijepa is not None:
            momentum = self.ema_momentum()
            self.net.ijepa.update_target(self.net.backbone, momentum)
            self.log("train/ema_momentum", momentum, on_step=True, on_epoch=False)

    def on_before_optimizer_step(self, optimizer):
        every = int(self.cfg["mlflow"].get("log_grad_norm_every_n_steps", 0) or 0)
        if not every or self.global_step % every:
            return
        norms = {}
        blocks = self.net.backbone.vit.blocks
        for index in range(self.net.backbone.first_trainable, len(blocks)):
            norms[f"grad_norm/backbone_block_{index:02d}"] = _grad_norm(blocks[index])
        norms["grad_norm/invariance_head"] = _grad_norm(self.net.invariance_head)
        norms["grad_norm/discrimination_head"] = _grad_norm(self.net.discrimination_head)
        if self.net.ijepa is not None:
            norms["grad_norm/ijepa_predictor"] = _grad_norm(self.net.ijepa.predictor)
        self.log_dict(norms, on_step=True, on_epoch=False)

    def configure_optimizers(self):
        cfg = self.cfg["optimizer"]
        groups = {}
        for name, parameter in self.net.named_parameters():
            if not parameter.requires_grad:
                continue
            part = "backbone" if name.startswith("backbone.") else "heads"
            decay = "no_decay" if parameter.ndim <= 1 or name.endswith(".bias") else "decay"
            groups.setdefault((part, decay), []).append(parameter)
        param_groups = [
            {
                "params": params,
                "lr": float(cfg["backbone_lr"] if part == "backbone" else cfg["head_lr"]),
                "weight_decay": float(cfg["weight_decay"]) if decay == "decay" else 0.0,
                "name": f"{part}_{decay}",
            }
            for (part, decay), params in groups.items()
        ]
        optimizer = torch.optim.AdamW(param_groups, betas=tuple(cfg.get("betas", (0.9, 0.999))))
        total = max(1, int(self.trainer.estimated_stepping_batches))
        warmup = int(round(float(cfg["warmup_epochs"]) * total / max(1, int(self.trainer.max_epochs))))
        floor = float(cfg.get("min_lr_ratio", 0.01))

        def schedule(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = min(1.0, (step - warmup) / max(1, total - warmup))
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    # --------------------------------------------------------------- validation
    def validation_step(self, batch, batch_idx):
        out = self.net(batch["image"])
        self._val_batches.append({
            "z_inv": out["z_inv"].float().cpu(),
            "z_dis": out["z_dis"].float().cpu(),
            "label": batch["label"].cpu(),
        })

    def on_validation_epoch_end(self):
        batches, self._val_batches = self._val_batches, []
        if self.trainer.sanity_checking or not batches:
            return
        z_inv = torch.cat([b["z_inv"] for b in batches]).numpy()
        z_dis = torch.cat([b["z_dis"] for b in batches]).numpy()
        labels = torch.cat([b["label"] for b in batches]).numpy()
        cluster_cfg = self.cfg["validation"]["clustering"]
        scorer = PairScorer(z_inv, z_dis, self.net.discrimination_head, use_energy=True)
        calibration, diag = fit_calibration(scorer, labels, cluster_cfg, seed=int(self.cfg["seed"]))
        best = diag["best"]
        same = diag["edge_same"].astype(bool)
        energy = -diag["edge_features"][:, 1]
        metrics = {
            "val/bcubed_f1": best["bcubed_f1"],
            "val/bcubed_precision": best["bcubed_precision"],
            "val/bcubed_recall": best["bcubed_recall"],
            "val/pairwise_f1": best["pairwise_f1"],
            "val/nmi": best["nmi"],
            "val/ari": best["ari"],
            "val/cluster_count_ratio": best["cluster_count_ratio"],
            "val/threshold": calibration["threshold"],
            "val/pair_auc": diag["pair_auc"],
            "val/pair_auc_cosine_only": diag["pair_auc_cosine_only"],
            "val/candidate_recall": diag["candidate_recall_of_positives"],
            "val/energy_pos": float(energy[same].mean()) if same.any() else 0.0,
            "val/energy_neg": float(energy[~same].mean()) if (~same).any() else 0.0,
            **{f"val/{k}": v for k, v in recall_at_k(diag["neighbors"], labels).items()},
        }
        if best["cluster_accuracy"] is not None:
            metrics["val/cluster_accuracy"] = best["cluster_accuracy"]
        self.log_dict({k: float(v) for k, v in metrics.items()}, on_epoch=True)
        self.calibration = calibration
        self.validation_diagnostics = {**diag, "labels": labels, "epoch": int(self.current_epoch)}

    # -------------------------------------------------------------- checkpoints
    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["config"] = self.cfg
        checkpoint["model_fingerprint"] = model_fingerprint(self.cfg)
        checkpoint["calibration"] = self.calibration
        logger = self.logger
        checkpoint["mlflow_run_id"] = getattr(logger, "run_id", None) if logger is not None else None

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        saved = checkpoint.get("model_fingerprint")
        if saved and saved != model_fingerprint(self.cfg):
            raise RuntimeError(
                "The checkpoint was trained with a different model section or image_size in train.yaml. "
                "Restore those settings, or set checkpoint.resume: never to start a new run."
            )
        self.calibration = checkpoint.get("calibration")


def _grad_norm(module: torch.nn.Module) -> float:
    squares = [p.grad.detach().float().norm() ** 2 for p in module.parameters() if p.grad is not None]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0
