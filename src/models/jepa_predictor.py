"""
JEPA Predictor for MI-JEPA.
Takes visible contextualized embeddings + learnable mask tokens,
predicts the target embeddings of masked instances.
"""
import torch
import torch.nn as nn


class JEPAPredictor(nn.Module):
    """
    Transformer-based predictor that predicts masked instance embeddings
    from visible ones.

    Args:
        embed_dim: Embedding dimension (384 for ViT-S).
        num_heads: Number of attention heads.
        num_layers: Number of decoder layers.
        ff_dim: Feed-forward hidden dim.
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
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        visible: torch.Tensor,
        num_masked: int,
    ) -> torch.Tensor:
        """
        Args:
            visible: (B, V, D) contextualized embeddings of visible instances.
            num_masked: Number of masked instances to predict.
        Returns:
            predicted: (B, M, D) predicted embeddings for masked instances.
        """
        B = visible.shape[0]

        # Expand mask tokens for the batch
        mask_tokens = self.mask_token.expand(B, num_masked, -1)

        # Decode: mask tokens query the visible set
        predicted = self.decoder(mask_tokens, visible)
        return self.norm(predicted)
