"""
Gated Attention Pooling (Ilse et al., 2018) for MI-JEPA.
Aggregates K instance embeddings into a single bag-level representation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionPooling(nn.Module):
    """
    Gated attention mechanism:
        a_k = softmax(w^T (tanh(V z_k) ⊙ σ(U z_k)))
        h_bag = Σ a_k z_k

    Args:
        embed_dim: Dimension of instance embeddings.
        attn_dim: Internal attention dimension.
    """

    def __init__(self, embed_dim: int = 384, attn_dim: int = 128):
        super().__init__()
        self.V = nn.Linear(embed_dim, attn_dim)
        self.U = nn.Linear(embed_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)

    def forward(
        self, z: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            z: (B, K, D) instance embeddings.
            return_attention: If True, also return attention weights.
        Returns:
            h_bag: (B, D) bag-level representation.
            attn_weights: (B, K) attention weights (if requested).
        """
        # (B, K, attn_dim)
        h_v = torch.tanh(self.V(z))
        h_u = torch.sigmoid(self.U(z))
        # (B, K, 1) → (B, K)
        attn_logits = self.w(h_v * h_u).squeeze(-1)
        attn_weights = F.softmax(attn_logits, dim=1)

        # Weighted sum: (B, K, 1) * (B, K, D) → (B, D)
        h_bag = (attn_weights.unsqueeze(-1) * z).sum(dim=1)

        if return_attention:
            return h_bag, attn_weights
        return h_bag, None
