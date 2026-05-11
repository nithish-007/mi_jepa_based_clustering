"""
Energy Head for MI-JEPA.
Scores instance-bag compatibility: E(z_i, c_bag) → scalar.
"""
import torch
import torch.nn as nn


class EnergyHead(nn.Module):
    """
    MLP that maps (instance_embedding, bag_context) → scalar energy.

    Low energy = instance compatible with bag (intra-bag).
    High energy = instance incompatible (inter-bag).

    Args:
        embed_dim: Dimension of instance embeddings.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(self, embed_dim: int = 384, hidden_dim: int = 128):
        super().__init__()
        # Input: [z_i; c_bag] → 2 * embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        instance_emb: torch.Tensor,
        bag_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            instance_emb: (B, K, D) or (B, D) instance embeddings.
            bag_context:  (B, K, D) or (B, D) bag context (excluding instance i).
        Returns:
            energy: (B, K) or (B,) scalar energy per instance.
        """
        combined = torch.cat([instance_emb, bag_context], dim=-1)
        return self.net(combined).squeeze(-1)
