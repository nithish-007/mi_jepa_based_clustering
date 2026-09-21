"""
MI-JEPA: Multi-Instance Joint Embedding Predictive Architecture.
Full dual-stream Lightning module.
"""
import math

import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightly.utils.scheduler import cosine_schedule

from src.models.encoders import ContextEncoder, TargetEncoder
from src.models.cross_instance_attention import CrossInstanceAttention
from src.models.jepa_predictor import JEPAPredictor
from src.models.energy_head import EnergyHead
from src.models.attention_pooling import GatedAttentionPooling
from src.losses.jepa_loss import DualJEPALoss
from src.losses.energy_loss import DualEnergyLoss
from src.losses.vicreg import VICRegLoss


class MIJEPAModule(pl.LightningModule):
    """
    Dual-stream MI-JEPA Lightning module.

    L = λ₁(L_JEPA^inter + L_JEPA^intra) + λ₂ L_energy + λ₃ L_reg

    Args:
        embed_dim: ViT embedding dimension.
        model_name: timm model name for backbone.
        pretrained: Use ImageNet pretrained backbone.
        bag_size: K instances per bag.
        mask_ratio: Fraction of instances to mask.
        cia_layers: Number of Cross-Instance Attention layers.
        cia_heads: Number of CIA attention heads.
        pred_layers: Number of JEPA predictor layers.
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        warmup_epochs: LR warmup epochs.
        max_epochs: Total training epochs.
        base_momentum: Starting EMA momentum.
        lambda_jepa: Weight for JEPA loss (λ₁).
        lambda_energy: Weight for energy loss (λ₂).
        lambda_reg: Weight for VICReg loss (λ₃).
        energy_margin_low: Low margin for energy loss.
        energy_margin_high: High margin for energy loss.
    """

    def __init__(
        self,
        embed_dim: int | None = 384,
        model_name: str = "vit_small_patch16_224",
        pretrained: bool = True,
        bag_size: int = 4,
        mask_ratio: float = 0.5,
        cia_layers: int = 2,
        cia_heads: int = 6,
        pred_layers: int = 2,
        lr: float = 1.5e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        base_momentum: float = 0.996,
        lambda_jepa: float = 1.0,
        lambda_energy: float = 0.0,
        lambda_reg: float = 0.1,
        energy_margin_low: float = 0.5,
        energy_margin_high: float = 2.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.bag_size = bag_size
        self.num_masked = max(1, int(bag_size * mask_ratio))
        self.num_visible = bag_size - self.num_masked

        # --- Model components ---
        # Backbone first — its `embed_dim` drives every downstream module so a
        # CLI/YAML mismatch can't desync the architecture.
        self.context_encoder = ContextEncoder(model_name, embed_dim, pretrained)
        embed_dim = self.context_encoder.embed_dim
        self.embed_dim = embed_dim

        self.target_encoder = TargetEncoder(self.context_encoder, base_momentum)
        self.cia = CrossInstanceAttention(
            embed_dim, cia_heads, cia_layers, ff_dim=embed_dim * 4
        )
        self.predictor = JEPAPredictor(
            embed_dim, cia_heads, pred_layers, ff_dim=embed_dim * 4
        )
        self.energy_head = EnergyHead(embed_dim)
        self.attention_pooling = GatedAttentionPooling(embed_dim)

        # --- Losses ---
        self.jepa_loss = DualJEPALoss(normalize_targets=True)
        self.energy_loss = DualEnergyLoss(energy_margin_low, energy_margin_high)
        self.vicreg_loss = VICRegLoss()

        # --- Hyperparams ---
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_momentum = base_momentum
        self.lambda_jepa = lambda_jepa
        self.lambda_energy = lambda_energy
        self.lambda_reg = lambda_reg

    def _encode_bag(self, bag: torch.Tensor) -> torch.Tensor:
        """
        Encode a bag of images through the context encoder.
        Args:
            bag: (B, K, C, H, W)
        Returns:
            z: (B, K, D)
        """
        B, K, C, H, W = bag.shape
        flat = bag.reshape(B * K, C, H, W)
        z = self.context_encoder(flat)  # (B*K, D)
        return z.reshape(B, K, -1)

    @torch.no_grad()
    def _encode_targets(self, bag: torch.Tensor) -> torch.Tensor:
        """Encode bag through the EMA target encoder (stop-gradient)."""
        B, K, C, H, W = bag.shape
        flat = bag.reshape(B * K, C, H, W)
        t = self.target_encoder(flat)
        return t.reshape(B, K, -1)

    def _mask_and_predict(
        self, h: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Random masking + JEPA prediction.
        Args:
            h: (B, K, D) contextualized embeddings.
            targets: (B, K, D) EMA target embeddings.
        Returns:
            predicted: (B, M, D) predictions for masked slots.
            masked_targets: (B, M, D) targets for masked slots.
        """
        B, K, D = h.shape
        M = self.num_masked

        # Random permutation to select which instances to mask
        perm = torch.randperm(K, device=h.device)
        visible_idx = perm[: K - M]
        masked_idx = perm[K - M :]

        visible = h[:, visible_idx]       # (B, V, D)
        masked_targets = targets[:, masked_idx]  # (B, M, D)

        predicted = self.predictor(visible, M)  # (B, M, D)
        return predicted, masked_targets

    def _compute_energy(
        self, z: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-instance energy scores.
        For each instance i, bag context = attention pool of other instances.
        Args:
            z: (B, K, D)
        Returns:
            energies: (B, K)
        """
        B, K, D = z.shape
        energies = []
        for i in range(K):
            # Context: all instances except i
            mask = torch.ones(K, dtype=torch.bool, device=z.device)
            mask[i] = False
            others = z[:, mask]  # (B, K-1, D)
            ctx, _ = self.attention_pooling(others)  # (B, D)
            e = self.energy_head(z[:, i], ctx)  # (B,)
            energies.append(e)
        return torch.stack(energies, dim=1)  # (B, K)

    def training_step(self, batch, batch_idx):
        inter_imgs, inter_labels, intra_imgs, intra_labels = batch

        # --- Encode both streams ---
        z_inter = self._encode_bag(inter_imgs)   # (B, K, D)
        z_intra = self._encode_bag(intra_imgs)   # (B, K, D)

        # --- EMA targets ---
        t_inter = self._encode_targets(inter_imgs)
        t_intra = self._encode_targets(intra_imgs)

        # --- Cross-Instance Attention ---
        h_inter = self.cia(z_inter)  # (B, K, D)
        h_intra = self.cia(z_intra)  # (B, K, D)

        # --- JEPA prediction ---
        pred_inter, tgt_inter = self._mask_and_predict(h_inter, t_inter)
        pred_intra, tgt_intra = self._mask_and_predict(h_intra, t_intra)

        jepa_total, jepa_inter, jepa_intra = self.jepa_loss(
            pred_inter, tgt_inter, pred_intra, tgt_intra
        )

        # --- VICReg regularization ---
        all_z = torch.cat([
            z_inter.reshape(-1, z_inter.shape[-1]),
            z_intra.reshape(-1, z_intra.shape[-1]),
        ], dim=0)
        reg_total, reg_var, reg_cov = self.vicreg_loss(all_z)

        # --- Total loss ---
        loss = self.lambda_jepa * jepa_total + self.lambda_reg * reg_total

        # --- Energy loss ---
        e_inter = self._compute_energy(z_inter)
        e_intra = self._compute_energy(z_intra)
        energy_total, energy_pos, energy_neg = self.energy_loss(e_intra, e_inter)
        loss = loss + self.lambda_energy * energy_total

        # --- EMA update ---
        self.target_encoder.update(self.context_encoder)

        # --- Logging ---
        self.log("train/loss", loss)
        self.log("train/jepa_total", jepa_total)
        self.log("train/jepa_inter", jepa_inter)
        self.log("train/jepa_intra", jepa_intra)
        self.log("train/reg_total", reg_total)
        self.log("train/reg_var", reg_var)
        self.log("train/reg_cov", reg_cov)
        self.log("train/energy_total", energy_total)
        self.log("train/energy_pos", energy_pos)
        self.log("train/energy_neg", energy_neg)
        self.log("train/energy_intra_mean", e_intra.mean())
        self.log("train/energy_inter_mean", e_inter.mean())

        return loss

    def validation_step(self, batch, batch_idx):
        inter_imgs, inter_labels, intra_imgs, intra_labels = batch

        z_inter = self._encode_bag(inter_imgs)
        z_intra = self._encode_bag(intra_imgs)

        t_inter = self._encode_targets(inter_imgs)
        t_intra = self._encode_targets(intra_imgs)

        h_inter = self.cia(z_inter)
        h_intra = self.cia(z_intra)

        pred_inter, tgt_inter = self._mask_and_predict(h_inter, t_inter)
        pred_intra, tgt_intra = self._mask_and_predict(h_intra, t_intra)

        jepa_total, jepa_inter, jepa_intra = self.jepa_loss(
            pred_inter, tgt_inter, pred_intra, tgt_intra
        )

        all_z = torch.cat([
            z_inter.reshape(-1, z_inter.shape[-1]),
            z_intra.reshape(-1, z_intra.shape[-1]),
        ], dim=0)
        reg_total, _, _ = self.vicreg_loss(all_z)

        loss = self.lambda_jepa * jepa_total + self.lambda_reg * reg_total

        e_inter = self._compute_energy(z_inter)
        e_intra = self._compute_energy(z_intra)
        energy_total, _, _ = self.energy_loss(e_intra, e_inter)
        loss = loss + self.lambda_energy * energy_total

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/jepa_total", jepa_total, sync_dist=True)
        self.log("val/energy_total", energy_total, sync_dist=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Ramp EMA momentum base_momentum -> 1.0 on a cosine schedule."""
        self.target_encoder.set_momentum(
            cosine_schedule(
                step=self.global_step,
                max_steps=max(self.trainer.estimated_stepping_batches, 1),
                start_value=self.base_momentum,
                end_value=1.0,
            )
        )

    def configure_optimizers(self):
        # Only optimize online encoder, CIA, predictor, energy head, attn pooling
        # (Target encoder is updated via EMA, not gradient)
        params = (
            list(self.context_encoder.parameters())
            + list(self.cia.parameters())
            + list(self.predictor.parameters())
            + list(self.energy_head.parameters())
            + list(self.attention_pooling.parameters())
        )
        optimizer = torch.optim.AdamW(
            params, lr=self.lr, weight_decay=self.weight_decay
        )

        # Linear warmup + cosine decay
        warmup_steps = (
            self.warmup_epochs * self.trainer.estimated_stepping_batches
            // self.max_epochs
        )
        total_steps = self.trainer.estimated_stepping_batches

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
