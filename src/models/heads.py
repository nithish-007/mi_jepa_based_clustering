"""The two parallel heads on the pooled backbone feature.

InvarianceHead: an embedding where all views of a product collapse together.
    Trained with supervised contrastive loss. Used for kNN retrieval.
DiscriminationHead: its own embedding plus an energy MLP E(a, b) that scores a pair.
    Low energy means same product. Trained with hard-negative margin ranking and BCE,
    so it focuses on fine differences (flavour, size, variant) the invariance head smooths over.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def mlp(in_dim: int, hidden_dims: list[int], out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = in_dim
    for hidden in hidden_dims:
        layers += [nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.GELU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        width = hidden
    layers.append(nn.Linear(width, out_dim))
    return nn.Sequential(*layers)


class InvarianceHead(nn.Module):
    def __init__(self, in_dim: int, cfg: dict):
        super().__init__()
        self.detach_input = bool(cfg.get("detach_input", False))
        self.net = mlp(in_dim, list(cfg["hidden_dims"]), int(cfg["out_dim"]), float(cfg.get("dropout", 0.0)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.detach_input:
            features = features.detach()
        return F.normalize(self.net(features).float(), dim=-1)


class DiscriminationHead(nn.Module):
    def __init__(self, in_dim: int, cfg: dict):
        super().__init__()
        self.detach_input = bool(cfg.get("detach_input", False))
        dim = int(cfg["out_dim"])
        self.embed = mlp(in_dim, list(cfg["hidden_dims"]), dim, float(cfg.get("dropout", 0.0)))
        self.energy_net = mlp(2 * dim + 1, list(cfg["energy_hidden_dims"]), 1, float(cfg.get("dropout", 0.0)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.detach_input:
            features = features.detach()
        return F.normalize(self.embed(features).float(), dim=-1)

    @staticmethod
    def pair_features(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Symmetric in (a, b), so E(a, b) == E(b, a) by construction."""
        return torch.cat([(a - b).abs(), a * b, (a * b).sum(dim=-1, keepdim=True)], dim=-1)

    def energy(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Energy for aligned rows of ``a`` and ``b``."""
        return self.energy_net(self.pair_features(a, b)).squeeze(-1).float()

    def energy_matrix(self, z: torch.Tensor) -> torch.Tensor:
        n = z.shape[0]
        a = z.unsqueeze(1).expand(n, n, -1)
        b = z.unsqueeze(0).expand(n, n, -1)
        return self.energy(a, b)
