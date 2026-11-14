import tempfile
import unittest
from pathlib import Path

import torch
from lightly.data.collate import IJEPAMaskCollator

from src.models.backbone import ViTBackbone
from src.models.network import ProductClusteringNet
from tests.fixtures import train_config

DINO = Path("~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth").expanduser()


def _masks(batch):
    collator = IJEPAMaskCollator(input_size=(64, 64), patch_size=8, pred_mask_scale=(0.15, 0.2),
                                 aspect_ratio=(0.75, 1.5), npred=4, enc_mask_scale=(0.85, 1.0), min_keep=2)
    _, context, targets = collator([torch.zeros(1)] * batch)
    return context, targets


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.config = train_config(Path(tempfile.gettempdir()))

    def test_only_last_blocks_are_trainable(self):
        backbone = ViTBackbone(self.config["model"]["backbone"], 64)
        trainable = [any(p.requires_grad for p in block.parameters()) for block in backbone.vit.blocks]
        self.assertEqual(trainable, [False, False, True, True])
        self.assertFalse(backbone.vit.patch_embed.proj.weight.requires_grad)
        self.assertFalse(backbone.vit.pos_embed.requires_grad)

    def test_ijepa_context_and_targets_are_disjoint(self):
        context, targets = _masks(6)
        for b in range(6):
            for target in targets:
                self.assertTrue(set(context[0][b].tolist()).isdisjoint(target[b].tolist()))

    def test_gradients_reach_trainable_parts_only_and_ema_moves_target(self):
        net = ProductClusteringNet(self.config["model"], 64)
        images = torch.randn(6, 3, 64, 64)
        context, targets = _masks(6)
        out = net(images)
        parts = net.ijepa(net.backbone, images, context, targets)
        energy = net.discrimination_head.energy_matrix(out["z_dis"])
        (parts["loss"] + out["z_inv"].sum() + energy.mean()).backward()
        self.assertIsNone(net.backbone.vit.blocks[0].attn.qkv.weight.grad)
        self.assertIsNotNone(net.backbone.vit.blocks[-1].attn.qkv.weight.grad)
        self.assertIsNotNone(net.ijepa.predictor.mask_token.grad)
        self.assertTrue(all(p.grad is None for p in net.ijepa.target_encoder.parameters()))

        optimizer = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=1.0)
        frozen_before = net.backbone.vit.blocks[0].attn.qkv.weight.clone()
        target_before = net.ijepa.target_encoder.vit.blocks[-1].attn.qkv.weight.clone()
        optimizer.step()
        net.ijepa.update_target(net.backbone, 0.5)
        online = net.backbone.vit.blocks[-1].attn.qkv.weight
        target = net.ijepa.target_encoder.vit.blocks[-1].attn.qkv.weight
        self.assertTrue(torch.equal(net.backbone.vit.blocks[0].attn.qkv.weight, frozen_before))
        self.assertTrue(torch.allclose(target, 0.5 * target_before + 0.5 * online))

    def test_energy_is_symmetric(self):
        net = ProductClusteringNet(self.config["model"], 64)
        energy = net.discrimination_head.energy_matrix(net.discrimination_head(torch.randn(5, 64)))
        self.assertTrue(torch.allclose(energy, energy.T, atol=1e-6))

    @unittest.skipUnless(DINO.is_file(), "local DINOv2 weights not present")
    def test_dinov2_weights_and_resampled_positions_load(self):
        cfg = {"name": "vit_small_patch14_dinov2.lvd142m", "checkpoint": str(DINO), "pretrained": True, "trainable_blocks": 4}
        backbone = ViTBackbone(cfg, 196)
        state = torch.load(DINO, map_location="cpu")
        self.assertEqual(backbone.grid, (14, 14))
        self.assertTrue(torch.equal(backbone.vit.blocks[0].attn.qkv.weight, state["blocks.0.attn.qkv.weight"]))
        self.assertTrue(torch.allclose(backbone.vit.pos_embed[0, 0], state["pos_embed"][0, 0]))
        self.assertGreater(backbone.vit.pos_embed[0, 1:].std(), 0.5 * state["pos_embed"][0, 1:].std())


if __name__ == "__main__":
    unittest.main()
