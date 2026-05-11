"""
Dual JEPA Loss for MI-JEPA.
Computes MSE between predicted and EMA-target embeddings for both inter and intra bags.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPALoss(nn.Module):
    """
    Cross-instance predictive loss.

    L_JEPA = (1/|M|) Σ_{m∈M} ||ĥ_m - sg(t_m)||²

    Args:
        use_smooth_l1: Use SmoothL1 instead of MSE (more stable).
        normalize_targets: L2-normalize targets before loss (I-JEPA style).
    """

    def __init__(
        self,
        use_smooth_l1: bool = False,
        normalize_targets: bool = True,
    ):
        super().__init__()
        self.use_smooth_l1 = use_smooth_l1
        self.normalize_targets = normalize_targets

    def forward(
        self,
        predicted: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predicted: (B, M, D) predicted embeddings for masked instances.
            targets: (B, M, D) EMA target embeddings (already stop-gradient).
        Returns:
            loss: scalar.
        """
        if self.normalize_targets:
            targets = F.normalize(targets, dim=-1)
            predicted = F.normalize(predicted, dim=-1)

        if self.use_smooth_l1:
            return F.smooth_l1_loss(predicted, targets)
        return F.mse_loss(predicted, targets)


class DualJEPALoss(nn.Module):
    """
    Combined JEPA loss for both inter and intra bags.
    L = L_JEPA^inter + L_JEPA^intra
    """

    def __init__(self, use_smooth_l1: bool = False, normalize_targets: bool = True):
        super().__init__()
        self.loss_fn = JEPALoss(use_smooth_l1, normalize_targets)

    def forward(
        self,
        pred_inter: torch.Tensor,
        targets_inter: torch.Tensor,
        pred_intra: torch.Tensor,
        targets_intra: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            total: combined loss.
            loss_inter: inter-bag JEPA loss.
            loss_intra: intra-bag JEPA loss.
        """
        loss_inter = self.loss_fn(pred_inter, targets_inter)
        loss_intra = self.loss_fn(pred_intra, targets_intra)
        return loss_inter + loss_intra, loss_inter, loss_intra
