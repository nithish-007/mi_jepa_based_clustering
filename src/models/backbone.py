"""Shared ViT backbone (timm) with a frozen prefix of blocks.

Only the last ``trainable_blocks`` transformer blocks (and the final norm) get
gradients. The frozen prefix runs under ``torch.no_grad``, so its activations are
not kept for backward. That is what makes ViT-S fit on a 4 GB GPU.
"""
from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from lightly.models.utils import apply_masks
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.config import resolve_path


def create_vit(cfg: dict, image_size: int) -> nn.Module:
    kwargs = {"num_classes": 0, "img_size": int(image_size), **(cfg.get("kwargs") or {})}
    pretrained = bool(cfg.get("pretrained", True))
    weights = resolve_path(cfg.get("checkpoint"))
    if pretrained and weights is not None:
        if not weights.is_file():
            raise FileNotFoundError(f"model.backbone.checkpoint {weights} does not exist")
        # timm resamples pos_embed to img_size when it loads through its pretrained path.
        kwargs["pretrained_cfg_overlay"] = {"file": str(weights)}
    return timm.create_model(cfg["name"], pretrained=pretrained, **kwargs)


class ViTBackbone(nn.Module):
    def __init__(self, cfg: dict, image_size: int):
        super().__init__()
        self.vit = create_vit(cfg, image_size)
        self.num_blocks = len(self.vit.blocks)
        self.embed_dim = int(self.vit.num_features)
        self.patch_size = int(self.vit.patch_embed.patch_size[0])
        self.grid = tuple(int(g) for g in self.vit.patch_embed.grid_size)
        self.num_patches = self.grid[0] * self.grid[1]
        self.num_prefix = int(getattr(self.vit, "num_prefix_tokens", 1))
        self.pooling = cfg.get("pooling", "cls")
        self.grad_checkpointing = bool(cfg.get("grad_checkpointing", False))
        self.set_trainable_blocks(int(cfg["trainable_blocks"]))

    def set_trainable_blocks(self, n: int) -> None:
        if not 0 <= n <= self.num_blocks:
            raise ValueError(f"trainable_blocks must be in [0, {self.num_blocks}], got {n}")
        self.first_trainable = self.num_blocks - n
        for parameter in self.vit.parameters():
            parameter.requires_grad = False
        for block in self.vit.blocks[self.first_trainable :]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        if n > 0:
            for parameter in self.vit.norm.parameters():
                parameter.requires_grad = True

    def embed_tokens(self, images: torch.Tensor, patch_masks=None) -> torch.Tensor:
        """Patch embedding plus positions, optionally keeping only ``patch_masks`` patches, plus prefix tokens."""
        vit = self.vit
        x = vit.patch_embed(images)
        if getattr(vit, "no_embed_class", False):
            patch_pos, prefix_pos = vit.pos_embed, None
        else:
            patch_pos, prefix_pos = vit.pos_embed[:, self.num_prefix :], vit.pos_embed[:, : self.num_prefix]
        x = x + patch_pos
        if patch_masks is not None:
            x = apply_masks(x, patch_masks)
        prefix = []
        if getattr(vit, "cls_token", None) is not None:
            prefix.append(vit.cls_token)
        if getattr(vit, "reg_token", None) is not None:
            prefix.append(vit.reg_token)
        if prefix:
            tokens = torch.cat(prefix, dim=1)
            if prefix_pos is not None:
                tokens = tokens + prefix_pos
            x = torch.cat([tokens.expand(x.shape[0], -1, -1), x], dim=1)
        return vit.norm_pre(vit.pos_drop(x))

    def forward_tokens(self, images: torch.Tensor, patch_masks=None) -> torch.Tensor:
        """Final-norm tokens ``(B, prefix + patches, D)``. ``patch_masks`` follows the I-JEPA convention."""
        x = self.embed_tokens(images, patch_masks)
        blocks = self.vit.blocks
        if self.first_trainable:
            with torch.no_grad():
                for block in blocks[: self.first_trainable]:
                    x = block(x)
        for block in blocks[self.first_trainable :]:
            if self.grad_checkpointing and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.vit.norm(x)

    def pool(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return tokens[:, 0]
        if self.pooling == "mean":
            return tokens[:, self.num_prefix :].mean(dim=1)
        raise ValueError(f"model.backbone.pooling must be cls or mean, got {self.pooling!r}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.pool(self.forward_tokens(images))

    @torch.no_grad()
    def block_outputs(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Tokens after every block (before the final norm). For layer-wise analysis."""
        x = self.embed_tokens(images)
        outputs = []
        for block in self.vit.blocks:
            x = block(x)
            outputs.append(x)
        return outputs


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)
