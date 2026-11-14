"""Checkpoint layout: ``checkpoints/<experiment.name>/``

    epoch_012-bcubed_f1_0.6123.ckpt   top-2 by val/bcubed_f1
    epoch_015-bcubed_f1_0.6150.ckpt
    last.ckpt                         end of the latest epoch, used to resume
"""
from __future__ import annotations

from pathlib import Path

import torch

from src.config import resolve_path

FILENAME = "epoch_{epoch:03d}-bcubed_f1_{val/bcubed_f1:.4f}"


def checkpoint_dir(config: dict) -> Path:
    return resolve_path(config["checkpoint"]["dir"]) / config["experiment"]["name"]


def best_checkpoint(directory: Path) -> Path:
    """The best path recorded by ModelCheckpoint in ``last.ckpt``."""
    last = directory / "last.ckpt"
    if not last.is_file():
        raise FileNotFoundError(f"{last} does not exist. Train first: python scripts/train.py")
    state = torch.load(last, map_location="cpu", weights_only=False)
    for key, value in state.get("callbacks", {}).items():
        if key.startswith("ModelCheckpoint") and value.get("best_model_path"):
            best = Path(value["best_model_path"])
            if not best.is_file():
                best = directory / best.name
            if best.is_file():
                return best
    raise FileNotFoundError(f"No best checkpoint recorded in {last}")


def resolve_checkpoint(spec: str | None, directory: Path) -> Path | None:
    """``best``, ``last``, an explicit path, or None (no checkpoint)."""
    if spec in (None, "", "none", "never"):
        return None
    if spec == "best":
        return best_checkpoint(directory)
    if spec in ("last", "auto"):
        last = directory / "last.ckpt"
        if spec == "auto":
            return last if last.is_file() else None
        if not last.is_file():
            raise FileNotFoundError(f"{last} does not exist")
        return last
    path = resolve_path(spec)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint {path} does not exist")
    return path
