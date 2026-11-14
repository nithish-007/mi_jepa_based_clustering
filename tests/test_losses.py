import unittest

import torch
import torch.nn.functional as F

from src.losses.discrimination import discrimination_loss
from src.losses.invariance import supervised_contrastive_loss


class LossTests(unittest.TestCase):
    def test_supcon_prefers_grouped_embeddings(self):
        torch.manual_seed(0)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        grouped = F.normalize(torch.eye(3)[labels] + 0.01 * torch.randn(6, 3), dim=1)
        shuffled = F.normalize(torch.randn(6, 3), dim=1)
        self.assertLess(supervised_contrastive_loss(grouped, labels, 0.1), supervised_contrastive_loss(shuffled, labels, 0.1))

    def test_supcon_without_positives_is_zero(self):
        z = F.normalize(torch.randn(4, 8), dim=1)
        self.assertEqual(supervised_contrastive_loss(z, torch.arange(4), 0.1).item(), 0.0)

    def test_discrimination_loss_rewards_separated_energies(self):
        labels = torch.tensor([0, 0, 1, 1])
        same = labels[:, None] == labels[None, :]
        good = torch.where(same, torch.tensor(-5.0), torch.tensor(5.0))
        good_out = discrimination_loss(good, labels, 1.0, 1.0, 1.0)
        bad_out = discrimination_loss(-good, labels, 1.0, 1.0, 1.0)
        self.assertEqual(good_out["ranking"].item(), 0.0)
        self.assertEqual(good_out["pair_accuracy"].item(), 1.0)
        self.assertGreater(bad_out["loss"], good_out["loss"])


if __name__ == "__main__":
    unittest.main()
