from __future__ import annotations

import torch
import torch.nn.functional as F


def discrimination_loss(energy: torch.Tensor, labels: torch.Tensor, margin: float,
                        ranking_weight: float, bce_weight: float) -> dict:
    """Energy objective on an in-batch ``(B, B)`` energy matrix. Low energy = same product.

    ranking: for each anchor, its hardest positive must have lower energy than its
        hardest negative by ``margin``.
    bce: every pair is a same/different example with logit ``-energy``; positive and
        negative terms are averaged separately because negatives outnumber positives.
    """
    n = energy.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=energy.device)
    same = labels[:, None] == labels[None, :]
    positives, negatives = same & ~eye, ~same

    hardest_pos = energy.masked_fill(~positives, float("-inf")).max(dim=1).values
    hardest_neg = energy.masked_fill(~negatives, float("inf")).min(dim=1).values
    valid = positives.any(dim=1) & negatives.any(dim=1)
    ranking = F.relu(margin + hardest_pos - hardest_neg)[valid].mean() if valid.any() else energy.sum() * 0.0

    upper = torch.triu(torch.ones_like(eye), diagonal=1)
    logits, targets = -energy[upper], same[upper].float()
    terms = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pos_term = terms[targets == 1].mean() if (targets == 1).any() else terms.sum() * 0.0
    neg_term = terms[targets == 0].mean() if (targets == 0).any() else terms.sum() * 0.0
    bce = 0.5 * (pos_term + neg_term)

    with torch.no_grad():
        stats = {
            "energy_pos": energy[positives].mean() if positives.any() else energy.new_tensor(0.0),
            "energy_neg": energy[negatives].mean(),
            "energy_hard_neg": hardest_neg[valid].mean() if valid.any() else energy.new_tensor(0.0),
            "pair_accuracy": ((logits > 0).float() == targets).float().mean(),
            "ranking_active": (margin + hardest_pos - hardest_neg > 0)[valid].float().mean() if valid.any() else energy.new_tensor(0.0),
        }
    return {"loss": ranking_weight * ranking + bce_weight * bce, "ranking": ranking, "bce": bce, **stats}
