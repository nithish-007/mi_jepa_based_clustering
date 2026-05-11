"""
VICReg-style anti-collapse regularization for MI-JEPA.
Prevents representation collapse via variance and covariance terms.
"""
import torch
import torch.nn as nn


class VICRegLoss(nn.Module):
    """
    Variance-Invariance-Covariance Regularization (Bardes et al., 2022).

    L_reg = μ_v · variance_loss + μ_c · covariance_loss

    Applied to the batch of instance embeddings to prevent collapse.

    Args:
        var_weight: Weight for variance term (μ_v).
        cov_weight: Weight for covariance term (μ_c).
        gamma: Target std for variance term.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.gamma = gamma
        self.eps = eps

    def variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Hinge loss on per-dimension std: encourage std >= gamma.
        z: (N, D)
        """
        std = torch.sqrt(z.var(dim=0) + self.eps)
        return torch.clamp(self.gamma - std, min=0).mean()

    def covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Penalize off-diagonal covariance: decorrelate dimensions.
        z: (N, D)
        """
        N, D = z.shape
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (N - 1)  # (D, D)

        # Zero out diagonal, penalize off-diagonal
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        return off_diag / D

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (N, D) instance embeddings (flattened across batch and bag).
        Returns:
            total: weighted sum.
            var_loss: variance component.
            cov_loss: covariance component.
        """
        var_loss = self.variance_loss(z)
        cov_loss = self.covariance_loss(z)
        total = self.var_weight * var_loss + self.cov_weight * cov_loss
        return total, var_loss, cov_loss
