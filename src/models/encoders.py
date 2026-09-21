"""
Context Encoder (ViT-Small or ResNet) and Target Encoder (EMA) for MI-JEPA.

The Context Encoder is backbone-agnostic: any timm model that produces a single
feature vector per image when created with `num_classes=0` is supported. The
output dimension is read directly from the timm model's `num_features` attribute
so callers don't have to keep the YAML `embed_dim` in sync with the backbone.
"""
import copy

import torch
import torch.nn as nn
import timm
from lightly.models.utils import deactivate_requires_grad, update_momentum


class ContextEncoder(nn.Module):
    """
    Image backbone that extracts a single embedding vector per image.

    Supported backbones (any timm model with a feature-vector head works):
      - "vit_small_patch16_224" -> 384-d  (CLS token)
      - "resnet18"              -> 512-d  (global avg pool)
      - "resnet50"              -> 2048-d
      - ...

    Args:
        model_name: timm model name.
        embed_dim: Expected output embedding dimension. If None, inferred from the
            backbone. If provided, asserted equal to the inferred dim (sanity check).
        pretrained: Use ImageNet pretrained weights.
        global_pool: Global pooling for conv backbones ('avg' for ResNet). Ignored
            by ViT (which uses CLS token).
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch16_224",
        embed_dim: int | None = 384,
        pretrained: bool = True,
        global_pool: str = "avg",
    ):
        super().__init__()
        # ViT models in timm ignore `global_pool="avg"` when CLS token is used; pass
        # it only for non-ViT models to avoid emitting harmless warnings.
        kwargs = {"num_classes": 0}
        if not model_name.startswith("vit"):
            kwargs["global_pool"] = global_pool

        self.backbone = timm.create_model(model_name, pretrained=pretrained, **kwargs)

        # `num_features` is the canonical timm attribute for the output dim of a
        # `num_classes=0` model.
        inferred = int(self.backbone.num_features)
        if embed_dim is not None and embed_dim != inferred:
            raise ValueError(
                f"embed_dim mismatch for backbone {model_name!r}: "
                f"configured={embed_dim}, inferred={inferred}. "
                f"Pass embed_dim=None to auto-infer or fix your config."
            )
        self.embed_dim = inferred

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

        # Frozen — updated only via EMA, never by the optimizer.
        deactivate_requires_grad(self.backbone)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def update(self, context_encoder: ContextEncoder):
        """EMA update from the context encoder."""
        update_momentum(context_encoder.backbone, self.backbone, m=self.momentum)

    def set_momentum(self, momentum: float):
        self.momentum = momentum
