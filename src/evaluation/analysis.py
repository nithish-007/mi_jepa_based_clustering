"""Diagnostics used by notebooks/analyze_model.ipynb.

Each function answers one question about a trained checkpoint, on the validation
split, and returns a DataFrame or arrays the notebook plots.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.evaluation.metrics import recall_at_k
from src.inference.clustering import knn
from src.losses.discrimination import discrimination_loss
from src.losses.invariance import supervised_contrastive_loss
from src.models.network import ProductClusteringNet


def load_network(checkpoint_path, device) -> tuple[ProductClusteringNet, dict, dict]:
    """Full network including the I-JEPA predictor and target encoder."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = state["config"]
    model_cfg = copy.deepcopy(config["model"])
    model_cfg["backbone"]["pretrained"] = False
    net = ProductClusteringNet(model_cfg, int(config["data"]["image_size"]))
    net.load_state_dict({k[len("net."):]: v for k, v in state["state_dict"].items() if k.startswith("net.")})
    return net.to(device).eval(), config, state


@torch.inference_mode()
def collect(net: ProductClusteringNet, loader, device, max_batches: int | None = None) -> dict:
    """Per-block CLS and mean-patch features plus head outputs for every image in ``loader``."""
    cls_by_block, mean_by_block, norms = [], [], []
    heads = {"pooled": [], "z_inv": [], "z_dis": []}
    labels = []
    prefix = net.backbone.num_prefix
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        images = batch["image"].to(device)
        blocks = net.backbone.block_outputs(images)
        cls_by_block.append(torch.stack([b[:, 0] for b in blocks]).float().cpu())
        mean_by_block.append(torch.stack([b[:, prefix:].mean(1) for b in blocks]).float().cpu())
        norms.append(torch.stack([b.float().norm(dim=-1).mean(1) for b in blocks]).cpu())
        out = net(images)
        for key in heads:
            heads[key].append(out[key].float().cpu())
        labels.append(batch["label"])
    return {
        "cls_by_block": torch.cat(cls_by_block, dim=1).numpy(),
        "mean_by_block": torch.cat(mean_by_block, dim=1).numpy(),
        "token_norm_by_block": torch.cat(norms, dim=1).numpy(),
        **{k: torch.cat(v).numpy() for k, v in heads.items()},
        "labels": torch.cat(labels).numpy(),
    }


def _recall1(features: np.ndarray, labels: np.ndarray) -> float:
    z = features / np.linalg.norm(features, axis=1, keepdims=True).clip(1e-12)
    neighbors, _ = knn(z, 1)
    return recall_at_k(neighbors, labels, ks=(1,))["recall_at_1"]


@torch.inference_mode()
def block_recall(backbone, loader, device) -> list[float]:
    """Per-block CLS kNN recall@1 for any backbone, e.g. the untouched pretrained one."""
    cls, labels = [], []
    for batch in loader:
        blocks = backbone.block_outputs(batch["image"].to(device))
        cls.append(torch.stack([b[:, 0] for b in blocks]).float().cpu())
        labels.append(batch["label"])
    cls, labels = torch.cat(cls, dim=1).numpy(), torch.cat(labels).numpy()
    return [_recall1(layer, labels) for layer in cls]


def layerwise_report(features: dict, first_trainable: int) -> pd.DataFrame:
    """kNN recall@1 of every block's CLS and mean-patch feature, and of each head output.

    If recall peaks before the last block, later blocks are hurting identity information.
    """
    labels = features["labels"]
    rows = []
    for index in range(features["cls_by_block"].shape[0]):
        rows.append({
            "layer": f"block_{index:02d}",
            "trainable": index >= first_trainable,
            "recall_at_1_cls": _recall1(features["cls_by_block"][index], labels),
            "recall_at_1_mean_patch": _recall1(features["mean_by_block"][index], labels),
            "mean_token_norm": float(features["token_norm_by_block"][index].mean()),
        })
    for head in ("pooled", "z_inv", "z_dis"):
        rows.append({"layer": head, "trainable": True, "recall_at_1_cls": _recall1(features[head], labels)})
    return pd.DataFrame(rows)


def per_loss_gradients(net: ProductClusteringNet, batch, config: dict, device) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gradient norm of each loss on each trainable block, and the cosine between loss gradients.

    A negative cosine means two objectives pull the shared blocks in opposite directions.
    """
    data, masks_context, masks_target = batch
    images, labels = data["image"].to(device), data["label"].to(device)
    masks_context = [m.to(device) for m in masks_context]
    masks_target = [m.to(device) for m in masks_target]
    loss_cfg = config["loss"]
    blocks = list(range(net.backbone.first_trainable, net.backbone.num_blocks))
    shared = [p for i in blocks for p in net.backbone.vit.blocks[i].parameters()]

    def compute(name: str) -> torch.Tensor:
        if name == "ijepa":
            return net.ijepa(net.backbone, images, masks_context, masks_target)["loss"]
        out = net(images)
        if name == "invariance":
            return supervised_contrastive_loss(out["z_inv"], labels, float(loss_cfg["invariance"]["temperature"]))
        d = loss_cfg["discrimination"]
        energy = net.discrimination_head.energy_matrix(out["z_dis"])
        return discrimination_loss(energy, labels, float(d["margin"]), float(d["ranking_weight"]), float(d["bce_weight"]))["loss"]

    names = ["invariance", "discrimination"] + (["ijepa"] if net.ijepa is not None else [])
    net.train()
    rows, flat = [], {}
    with torch.enable_grad():
        for name in names:
            net.zero_grad(set_to_none=True)
            value = compute(name) * float(loss_cfg[name]["weight"])
            value.backward()
            row = {"loss": name, "value": float(value.detach())}
            for i in blocks:
                grads = [p.grad.norm() ** 2 for p in net.backbone.vit.blocks[i].parameters() if p.grad is not None]
                row[f"block_{i:02d}"] = float(torch.stack(grads).sum().sqrt()) if grads else 0.0
            rows.append(row)
            flat[name] = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in shared])
    net.zero_grad(set_to_none=True)
    net.eval()
    names = list(flat)
    cosine = pd.DataFrame([[float(F.cosine_similarity(flat[a], flat[b], dim=0)) for b in names] for a in names], index=names, columns=names)
    return pd.DataFrame(rows), cosine


def pair_report(z_inv: np.ndarray, z_dis: np.ndarray, labels: np.ndarray, head, k: int = 10, seed: int = 0) -> dict:
    """Cosine and energy for positives, random negatives and hard negatives (nearest other product)."""
    rng = np.random.default_rng(seed)
    neighbors, _ = knn(z_inv, k)
    n = len(labels)
    anchors, positives, hard = [], [], []
    for i in range(n):
        same = np.flatnonzero((labels == labels[i]) & (np.arange(n) != i))
        other = neighbors[i][labels[neighbors[i]] != labels[i]]
        if len(same) and len(other):
            anchors.append(i)
            positives.append(rng.choice(same))
            hard.append(other[0])
    anchors, positives, hard = map(np.array, (anchors, positives, hard))
    random_neg = rng.integers(0, n, len(anchors))
    keep = labels[random_neg] != labels[anchors]
    head = copy.deepcopy(head).cpu().float().eval()
    zd = torch.as_tensor(z_dis)

    def score(j):
        with torch.inference_mode():
            energy = head.energy(zd[anchors], zd[j]).numpy()
        return (z_inv[anchors] * z_inv[j]).sum(1), energy

    report = {}
    for name, partner, mask in (("positive", positives, slice(None)), ("hard_negative", hard, slice(None)), ("random_negative", random_neg, keep)):
        cos, energy = score(partner)
        report[name] = {"cosine": cos[mask], "energy": energy[mask]}
    return report


def diagnose(layers: pd.DataFrame, grads: pd.DataFrame, grad_cosine: pd.DataFrame, calibration_diag: dict,
             pairs: dict, ijepa_stats: dict | None, margin: float) -> pd.DataFrame:
    """Turn measurements into concrete train.yaml / evaluate.yaml edits for the next run."""
    findings = []

    def add(observation, value, suggestion):
        findings.append({"observation": observation, "measured": value, "what to change": suggestion})

    blocks = layers[layers["layer"].str.startswith("block_")]
    best = blocks.loc[blocks["recall_at_1_cls"].idxmax()]
    last = blocks.iloc[-1]
    if best["layer"] != last["layer"] and best["recall_at_1_cls"] - last["recall_at_1_cls"] > 0.02:
        add("kNN recall peaks before the last block", f"{best['layer']}={best['recall_at_1_cls']:.3f} vs last={last['recall_at_1_cls']:.3f}",
            "lower optimizer.backbone_lr, or reduce model.backbone.trainable_blocks")
    z_inv = float(layers.loc[layers["layer"] == "z_inv", "recall_at_1_cls"].iloc[0])
    pooled = float(layers.loc[layers["layer"] == "pooled", "recall_at_1_cls"].iloc[0])
    if z_inv < pooled - 0.02:
        add("invariance head loses identity information vs the pooled feature", f"z_inv={z_inv:.3f}, pooled={pooled:.3f}",
            "widen model.invariance_head.hidden_dims / out_dim, or raise loss.invariance.temperature")
    if not blocks["trainable"].iloc[0] and blocks.loc[blocks["trainable"], "recall_at_1_cls"].max() <= blocks.loc[~blocks["trainable"], "recall_at_1_cls"].max():
        add("trainable blocks do not improve on the frozen prefix", f"best frozen={blocks.loc[~blocks['trainable'], 'recall_at_1_cls'].max():.3f}",
            "raise model.backbone.trainable_blocks (e.g. 6) or optimizer.backbone_lr")

    recall = calibration_diag["candidate_recall_of_positives"]
    if recall < 0.8:
        add("kNN misses many same-product pairs before scoring", f"{recall:.3f}", "raise validation.clustering.k_neighbors / clustering.k_neighbors")
    gain = calibration_diag["pair_auc"] - calibration_diag["pair_auc_cosine_only"]
    if gain < 0.005:
        add("energy adds little over cosine for pair verification", f"AUC gain {gain:+.4f}",
            "raise loss.discrimination.weight or data.sampler.same_family_prob; check discrimination_head size")
    hard_gap = float(np.mean(pairs["hard_negative"]["energy"]) - np.mean(pairs["positive"]["energy"]))
    if hard_gap < margin:
        add("hard negatives sit close to positives in energy", f"mean gap {hard_gap:.3f}",
            "raise data.sampler.same_family_prob or loss.discrimination.margin")
    sweep = calibration_diag["best"]
    if sweep["bcubed_precision"] - sweep["bcubed_recall"] > 0.15:
        add("clusters are pure but fragmented", f"P={sweep['bcubed_precision']:.3f} R={sweep['bcubed_recall']:.3f}",
            "add lower values to threshold_grid or use linkage_check: single")
    elif sweep["bcubed_recall"] - sweep["bcubed_precision"] > 0.15:
        add("clusters merge different products", f"P={sweep['bcubed_precision']:.3f} R={sweep['bcubed_recall']:.3f}",
            "add higher values to threshold_grid; keep linkage_check: average")

    if len(grads) > 1:
        cols = [c for c in grads.columns if c.startswith("block_")]
        totals = grads.set_index("loss")[cols].sum(axis=1)
        ratio = totals.max() / max(totals.min(), 1e-12)
        if ratio > 10:
            add("one loss dominates the backbone gradient", f"{totals.idxmax()} is {ratio:.0f}x {totals.idxmin()}",
                f"lower loss.{totals.idxmax()}.weight or raise loss.{totals.idxmin()}.weight")
        off = grad_cosine.where(~np.eye(len(grad_cosine), dtype=bool)).stack()
        if len(off) and off.min() < -0.1:
            a, b = off.idxmin()
            add("two losses pull the shared blocks in opposite directions", f"cos({a},{b})={off.min():.2f}",
                f"stage them with loss.{a}.start_epoch / loss.{b}.start_epoch, or detach_input on one head")
    if ijepa_stats is not None and ijepa_stats["target_std"] < 0.05:
        add("I-JEPA targets have very low spread (possible collapse)", f"std {ijepa_stats['target_std']:.4f}",
            "lower optimizer.head_lr, or raise model.ijepa.ema.start (e.g. 0.998)")
    if not findings:
        add("no rule fired", "-", "compare val/bcubed_f1 across runs in MLflow before changing anything")
    return pd.DataFrame(findings)
