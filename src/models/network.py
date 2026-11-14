"""Full model: shared backbone -> {I-JEPA branch, invariance head, discrimination head}.

                   +--> I-JEPA predictor (+ EMA target encoder)    [training only]
    image -> ViT --+--> pooled -> InvarianceHead      -> z_inv    (kNN candidates)
                   +--> pooled -> DiscriminationHead  -> z_dis -> E(z_dis_i, z_dis_j)
"""
from __future__ import annotations

import torch
from torch import nn

from src.models.backbone import ViTBackbone
from src.models.heads import DiscriminationHead, InvarianceHead
from src.models.ijepa import IJEPABranch


class ProductClusteringNet(nn.Module):
    def __init__(self, model_cfg: dict, image_size: int):
        super().__init__()
        self.backbone = ViTBackbone(model_cfg["backbone"], image_size)
        dim = self.backbone.embed_dim
        self.invariance_head = InvarianceHead(dim, model_cfg["invariance_head"])
        self.discrimination_head = DiscriminationHead(dim, model_cfg["discrimination_head"])
        self.ijepa = IJEPABranch(self.backbone, model_cfg["ijepa"]) if model_cfg["ijepa"].get("enabled", True) else None

    def forward(self, images: torch.Tensor) -> dict:
        pooled = self.backbone(images)
        return {
            "pooled": pooled,
            "z_inv": self.invariance_head(pooled),
            "z_dis": self.discrimination_head(pooled),
        }

    def energy(self, z_dis_a: torch.Tensor, z_dis_b: torch.Tensor) -> torch.Tensor:
        return self.discrimination_head.energy(z_dis_a, z_dis_b)

    def parameter_table(self) -> list[dict]:
        rows = []
        for name, module in [("backbone", self.backbone), ("invariance_head", self.invariance_head),
                             ("discrimination_head", self.discrimination_head)]:
            rows.append(_count(name, module))
        if self.ijepa is not None:
            rows.append(_count("ijepa_predictor", self.ijepa.predictor))
            rows.append(_count("ijepa_target_encoder (EMA)", self.ijepa.target_encoder))
        return rows


def _count(name: str, module: nn.Module) -> dict:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"module": name, "parameters": total, "trainable": trainable}
