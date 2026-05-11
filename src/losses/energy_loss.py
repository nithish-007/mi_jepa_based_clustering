"""
Dual Energy-Based Contrastive Loss for MI-JEPA.
Intra-bags → LOW energy (compatible instances).
Inter-bags → HIGH energy (incompatible instances).
"""
import torch
import torch.nn as nn


class DualEnergyLoss(nn.Module):
    """
    Margin-based energy loss using dual-stream bags.

    L = Σ max(0, E_intra - m_low)² + Σ max(0, m_high - E_inter)²

    Intra-bags provide positive signal (low energy target).
    Inter-bags provide negative signal (high energy target).

    Args:
        margin_low: Target upper bound for intra-bag energy.
        margin_high: Target lower bound for inter-bag energy.
    """

    def __init__(self, margin_low: float = 0.5, margin_high: float = 2.0):
        super().__init__()
        self.margin_low = margin_low
        self.margin_high = margin_high

    def forward(
        self,
        energy_intra: torch.Tensor,
        energy_inter: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            energy_intra: (B, K) energy scores for intra-bag instances.
            energy_inter: (B, K) energy scores for inter-bag instances.
        Returns:
            total: combined loss.
            loss_pos: positive (intra) component.
            loss_neg: negative (inter) component.
        """
        # Intra — push energy below margin_low
        loss_pos = torch.clamp(energy_intra - self.margin_low, min=0).pow(2).mean()

        # Inter — push energy above margin_high
        loss_neg = torch.clamp(self.margin_high - energy_inter, min=0).pow(2).mean()

        return loss_pos + loss_neg, loss_pos, loss_neg


class InfoNCEEnergyLoss(nn.Module):
    """
    InfoNCE-style energy loss (optional upgrade over margin-based).

    L = -log( Σ exp(-E_intra/τ) / (Σ exp(-E_intra/τ) + Σ exp(-E_inter/τ)) )
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        energy_intra: torch.Tensor,
        energy_inter: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            energy_intra: (B, K) energy for intra-bag (positive).
            energy_inter: (B, K) energy for inter-bag (negative).
        Returns:
            loss: scalar.
        """
        # Flatten to (B*K,)
        pos = (-energy_intra / self.temperature).reshape(-1)
        neg = (-energy_inter / self.temperature).reshape(-1)

        # log-sum-exp for numerical stability
        pos_logsumexp = torch.logsumexp(pos, dim=0)
        all_logsumexp = torch.logsumexp(torch.cat([pos, neg]), dim=0)

        return -(pos_logsumexp - all_logsumexp)
