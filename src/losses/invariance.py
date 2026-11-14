from __future__ import annotations

import torch


def supervised_contrastive_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """SupCon (Khosla et al., 2020) on L2-normalised embeddings. Anchors without a positive are skipped."""
    z = z.float()
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    logits = (z @ z.T / temperature).masked_fill(eye, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positives = (labels[:, None] == labels[None, :]) & ~eye
    count = positives.sum(dim=1)
    has_positive = count > 0
    if not has_positive.any():
        return z.sum() * 0.0
    summed = log_prob.masked_fill(~positives, 0.0).sum(dim=1)
    return -(summed[has_positive] / count[has_positive]).mean()
