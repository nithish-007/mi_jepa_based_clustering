from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler


class PKBatchSampler(Sampler[list[int]]):
    """Batches of P products x K images, so every batch has positives and negatives.

    With probability ``same_family_prob`` the next product is drawn from the same
    family as an already chosen product (size variants of one brand), which gives
    the discrimination head hard negatives. Products with fewer than K images are
    sampled with replacement, so their positives are two augmentations.
    Lightning calls ``set_epoch`` every epoch, so the order is reproducible and resumable.
    """

    def __init__(self, labels, families, products_per_batch: int, images_per_product: int,
                 batches_per_epoch: int | None = None, same_family_prob: float = 0.0, seed: int = 0):
        self.labels = np.asarray(labels)
        self.P, self.K = int(products_per_batch), int(images_per_product)
        self.same_family_prob = float(same_family_prob)
        self.seed, self.epoch = int(seed), 0
        self.by_label = {int(c): np.flatnonzero(self.labels == c) for c in np.unique(self.labels)}
        self.classes = np.array(sorted(self.by_label))
        if len(self.classes) < self.P:
            raise ValueError(f"Need at least {self.P} products for a batch, found {len(self.classes)}")
        family_of = dict(zip(self.labels.tolist(), np.asarray(families).tolist()))
        groups: dict = {}
        for c in self.classes:
            groups.setdefault(family_of[int(c)], []).append(int(c))
        self.siblings = {c: [o for o in groups[family_of[c]] if o != c] for c in map(int, self.classes)}
        self.n_batches = int(batches_per_epoch) if batches_per_epoch else max(1, len(self.labels) // (self.P * self.K))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n_batches

    def _choose_products(self, rng) -> list[int]:
        chosen: list[int] = []
        while len(chosen) < self.P:
            if chosen and rng.random() < self.same_family_prob:
                options = [s for c in chosen for s in self.siblings[c] if s not in chosen]
                if options:
                    chosen.append(int(rng.choice(options)))
                    continue
            candidate = int(rng.choice(self.classes))
            if candidate not in chosen:
                chosen.append(candidate)
        return chosen

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        for _ in range(self.n_batches):
            batch = []
            for c in self._choose_products(rng):
                pool = self.by_label[c]
                batch.extend(rng.choice(pool, self.K, replace=len(pool) < self.K).tolist())
            yield batch
