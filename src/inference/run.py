"""Cluster a set of images with a trained model and write one folder per cluster."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import resolve_path
from src.data.dataset import encode_labels
from src.data.indexing import list_image_files
from src.data.splits import load_manifest, select_records
from src.evaluation.metrics import clustering_metrics
from src.inference.embedding import extract_embeddings, load_model, pick_device
from src.inference.export import export_clusters
from src.inference.pipeline import GRAPH_METHODS, calibrate, predict, supports


def _inputs(config: dict) -> tuple[list[Path], np.ndarray | None, str]:
    source = config["input"]
    if source.get("image_dir"):
        directory = resolve_path(source["image_dir"])
        paths = list_image_files(directory)
        if not paths:
            raise FileNotFoundError(f"No images under {directory}")
        return paths, None, str(directory)
    frame = select_records(load_manifest(source["manifest"]), source["split"], source.get("max_skus"), source.get("max_images_per_sku"))
    root = resolve_path(source["data_root"])
    labels, _ = encode_labels(frame["sku_id"])
    return [root / p for p in frame["image_path"]], labels, f"{source['manifest']}:{source['split']}"


def _calibration(config: dict, model, method: str, device) -> tuple[dict | None, str]:
    if method not in GRAPH_METHODS:
        return None, "not needed"
    spec = config.get("calibration", "auto")
    if spec == "auto":
        saved = resolve_path(config["export"]["output_dir"]) / f"evaluation_{model.name}" / "calibration.json"
        if saved.is_file():
            stored = json.loads(saved.read_text(encoding="utf-8"))
            if method in stored:
                return stored[method], str(saved)
        if method == "graph" and model.calibration:
            return model.calibration, f"{model.checkpoint} (validation at that epoch)"
        spec = "validation"
    if spec == "validation":
        source = config["input"]
        val = select_records(load_manifest(source["manifest"]), "val", None, config.get("calibration_max_images_per_sku", 10))
        root = resolve_path(source["data_root"])
        emb = extract_embeddings(model, [root / p for p in val["image_path"]], int(config["embedding"]["batch_size"]),
                                 int(config["embedding"]["num_workers"]), device)
        labels, _ = encode_labels(val["sku_id"])
        calibration, _ = calibrate(method, model, emb, labels, config["clustering"], int(config["seed"]))
        return calibration, "fitted now on the validation split"
    path = resolve_path(spec)
    return json.loads(path.read_text(encoding="utf-8"))[method], str(path)


def run_clustering(config: dict) -> dict:
    device = pick_device(config.get("device", "auto"))
    model = load_model(config["model"], config["checkpoint_root"], device)
    method = config["method"]
    if not supports(method, model):
        raise ValueError(f"{model.name} has no discrimination head; use method graph_cosine or hdbscan")
    paths, labels, source = _inputs(config)
    calibration, calibration_source = _calibration(config, model, method, device)
    print(f"Embedding {len(paths)} images with {model.name} ({model.checkpoint})", flush=True)
    embeddings = extract_embeddings(model, paths, int(config["embedding"]["batch_size"]), int(config["embedding"]["num_workers"]), device)
    pred, extra = predict(method, model, embeddings, config["clustering"], int(config["seed"]), calibration)
    meta = {
        "model": model.name,
        "checkpoint": str(model.checkpoint),
        "method": method,
        "input": source,
        "calibration_source": calibration_source,
        "threshold": None if calibration is None else calibration.get("threshold"),
    }
    if labels is not None and config.get("score_if_labels", True):
        meta["metrics_against_sku_ids"] = clustering_metrics(labels, pred)
    output_dir = resolve_path(config["export"]["output_dir"]) / f"cluster_{model.name}"
    return export_clusters(paths, pred, extra, config["export"], output_dir, meta)
