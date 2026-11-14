"""I-JEPA branch (Assran et al., CVPR 2023): the invariance-by-prediction pathway.

- Context encoder = the shared online backbone, fed only the context patches.
- Target encoder = an EMA copy of the backbone, never in the optimizer, run under no_grad.
- Predictor = Lightly's port of the official narrow ViT predictor. It reads the
  context tokens plus mask tokens at the target positions.
- Loss = smooth L1 between predictions and layer-normed target patch features.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from lightly.models.modules.ijepa_timm import IJEPAPredictorTIMM
from lightly.models.utils import apply_masks, repeat_interleave_batch, update_momentum
from torch import nn

from src.models.backbone import ViTBackbone


class IJEPABranch(nn.Module):
    def __init__(self, backbone: ViTBackbone, cfg: dict):
        super().__init__()
        predictor_cfg = cfg["predictor"]
        self.predictor = IJEPAPredictorTIMM(
            num_patches=backbone.num_patches,
            depth=int(predictor_cfg["depth"]),
            mlp_dim=backbone.embed_dim,
            predictor_embed_dim=int(predictor_cfg["embed_dim"]),
            num_heads=int(predictor_cfg["num_heads"]),
        )
        self.target_encoder = copy.deepcopy(backbone)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False
        self.num_prefix = backbone.num_prefix

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target(self, backbone: ViTBackbone, momentum: float) -> None:
        update_momentum(backbone, self.target_encoder, momentum)

    def forward(self, backbone: ViTBackbone, images: torch.Tensor, masks_context: list, masks_target: list) -> dict:
        batch = images.shape[0]
        with torch.no_grad():
            target = self.target_encoder.forward_tokens(images)[:, self.num_prefix :]
            target = F.layer_norm(target, (target.shape[-1],))
            target = apply_masks(target, masks_target)
            target = repeat_interleave_batch(target, batch, repeat=len(masks_context))
        context = backbone.forward_tokens(images, patch_masks=masks_context)[:, self.num_prefix :]
        predicted = self.predictor(context, masks_context, masks_target)
        loss = F.smooth_l1_loss(predicted.float(), target.float())
        with torch.no_grad():
            cosine = F.cosine_similarity(predicted.float(), target.float(), dim=-1).mean()
            target_std = target.float().std(dim=0).mean()
        return {
            "loss": loss,
            "pred_cosine": cosine,
            "target_std": target_std,
            "context_patches": torch.tensor(float(context.shape[1])),
            "target_patches": torch.tensor(float(target.shape[1])),
        }
