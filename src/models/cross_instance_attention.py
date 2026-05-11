"""
Cross-Instance Attention (CIA) module for MI-JEPA.
2-layer Transformer Encoder operating across K instance embeddings within a bag.
No positional encoding — bags are unordered sets.
"""
import torch
import torch.nn as nn


class CrossInstanceAttention(nn.Module):
    """
    Transformer encoder that contextualizes instance embeddings within a bag.

    Args:
        embed_dim: Dimension of instance embeddings (384 for ViT-S).
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        ff_dim: Feed-forward hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 2,
        ff_dim: int = 1536,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, K, D) instance embeddings for a bag of K instances.
        Returns:
            h: (B, K, D) contextualized instance embeddings.
        """
        return self.norm(self.encoder(x))
