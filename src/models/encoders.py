"""
Context Encoder (ViT-Small) and Target Encoder (EMA) for MI-JEPA.
"""
import copy
import math

import torch
import torch.nn as nn
import timm


class ContextEncoder(nn.Module):
    """
    ViT-Small backbone. Extracts [CLS] token as instance embedding.

    Args:
        model_name: timm model name.
        embed_dim: Output embedding dimension (384 for ViT-Small).
        pretrained: Use ImageNet pretrained weights.
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch16_224",
        embed_dim: int = 384,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) images.
        Returns:
            z: (N, embed_dim) instance embeddings.
        """
        return self.backbone(x)


class TargetEncoder(nn.Module):
    """
    EMA (Exponential Moving Average) copy of the context encoder.
    Updated via momentum: θ̄ ← τ·θ̄ + (1-τ)·θ

    Args:
        context_encoder: The online context encoder to copy.
        initial_momentum: Starting EMA momentum (e.g. 0.996).
    """

    def __init__(self, context_encoder: ContextEncoder, initial_momentum: float = 0.996):
        super().__init__()
        self.backbone = copy.deepcopy(context_encoder.backbone)
        self.embed_dim = context_encoder.embed_dim
        self.momentum = initial_momentum

        # Freeze parameters — updated only via EMA
        for p in self.backbone.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @torch.no_grad()
    def update(self, context_encoder: ContextEncoder):
        """EMA update from the context encoder."""
        for p_target, p_online in zip(
            self.backbone.parameters(), context_encoder.backbone.parameters()
        ):
            p_target.data.mul_(self.momentum).add_(
                p_online.data, alpha=1.0 - self.momentum
            )

    def set_momentum(self, momentum: float):
        self.momentum = momentum


def cosine_momentum_schedule(
    base_momentum: float, total_steps: int
) -> list[float]:
    """
    Cosine schedule for EMA momentum: base_momentum → 1.0.
    Returns a list of momentum values, one per step.
    """
    return [
        1.0 - (1.0 - base_momentum) * (math.cos(math.pi * t / total_steps) + 1) / 2
        for t in range(total_steps)
    ]
