"""Load a trained model (or the frozen-backbone baseline) and embed images."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from src.config import resolve_path
from src.data.dataset import ProductImageDataset
from src.data.datamodule import loader_kwargs
from src.data.transforms import eval_transform
from src.models.backbone import ViTBackbone
from src.models.network import ProductClusteringNet
from src.training.checkpoints import resolve_checkpoint


class FrozenBackboneBaseline(nn.Module):
    """Pretrained backbone with no training. ``z_inv`` is the normalised pooled feature; no energy head."""

    def __init__(self, backbone_cfg: dict, image_size: int):
        super().__init__()
        self.backbone = ViTBackbone({**backbone_cfg, "trainable_blocks": 0}, image_size)
        self.discrimination_head = None

    def forward(self, images):
        pooled = self.backbone(images)
        return {"pooled": pooled, "z_inv": F.normalize(pooled.float(), dim=-1), "z_dis": None}


@dataclass
class LoadedModel:
    name: str
    net: nn.Module
    data_cfg: dict
    checkpoint: Path | None
    calibration: dict | None
    train_config: dict | None

    @property
    def has_energy(self) -> bool:
        return getattr(self.net, "discrimination_head", None) is not None


def load_model(entry: dict, checkpoint_root: str, device: torch.device) -> LoadedModel:
    """``entry`` = {name, checkpoint: best | last | <path> | null, [backbone, data] for the baseline}."""
    name = entry["name"]
    path = resolve_checkpoint(entry.get("checkpoint"), resolve_path(checkpoint_root) / name)
    if path is None:
        data_cfg = entry["data"]
        net = FrozenBackboneBaseline(entry["backbone"], int(data_cfg["image_size"]))
        return LoadedModel(name, net.to(device).eval(), data_cfg, None, None, None)
    state = torch.load(path, map_location="cpu", weights_only=False)
    config = state["config"]
    model_cfg = copy.deepcopy(config["model"])
    model_cfg["backbone"]["pretrained"] = False
    model_cfg["ijepa"]["enabled"] = False
    net = ProductClusteringNet(model_cfg, int(config["data"]["image_size"]))
    weights = {k[len("net."):]: v for k, v in state["state_dict"].items() if k.startswith("net.") and not k.startswith("net.ijepa.")}
    net.load_state_dict(weights, strict=True)
    return LoadedModel(name, net.to(device).eval(), config["data"], path, state.get("calibration"), config)


@torch.inference_mode()
def extract_embeddings(model: LoadedModel, paths: list[Path], batch_size: int, num_workers: int,
                       device: torch.device) -> dict[str, np.ndarray]:
    dataset = ProductImageDataset(paths, None, eval_transform(model.data_cfg))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **loader_kwargs(num_workers))
    outputs: dict[str, list] = {"pooled": [], "z_inv": [], "z_dis": []}
    use_amp = device.type == "cuda"
    for batch in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            out = model.net(batch["image"].to(device, non_blocking=True))
        for key in outputs:
            if out.get(key) is not None:
                outputs[key].append(out[key].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in outputs.items() if v}


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
