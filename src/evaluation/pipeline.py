"""Evaluate models on SKU-disjoint test products.

Per model: calibrate the pair scorer on the validation split, then cluster each test
corpus with every method and score it against the true SKU ids. Results go to
``results/evaluation_<model>/`` and one row per (model, corpus, method) to
``results/evaluation_summary.csv``. Test labels are used only for scoring.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.config import resolve_path, save_yaml
from src.data.dataset import encode_labels
from src.data.splits import load_manifest, select_records
from src.evaluation.metrics import clustering_metrics, recall_at_k
from src.evaluation.plots import cluster_size_figure, pair_diagnostics_figure
from src.inference.embedding import extract_embeddings, load_model, pick_device
from src.inference.pipeline import GRAPH_METHODS, calibrate, predict, supports


def _embed_frame(model, frame: pd.DataFrame, root: Path, cfg: dict, device) -> dict:
    paths = [root / p for p in frame["image_path"]]
    return extract_embeddings(model, paths, int(cfg["batch_size"]), int(cfg["num_workers"]), device)


def _subset(embeddings: dict, rows: np.ndarray) -> dict:
    return {k: v[rows] for k, v in embeddings.items()}


def evaluate_model(entry: dict, config: dict, manifest: pd.DataFrame, device) -> list[dict]:
    root = resolve_path(config["data"]["root"])
    out_dir = resolve_path(config["output_dir"]) / f"evaluation_{entry['name']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(entry, config["checkpoint_root"], device)
    seed, cluster_cfg = int(config["seed"]), config["clustering"]
    methods = [m for m in config["methods"] if supports(m, model)]
    print(f"[{model.name}] checkpoint={model.checkpoint} methods={methods}", flush=True)

    cal_cfg = config["calibration"]
    val = select_records(manifest, cal_cfg["split"], cal_cfg.get("max_skus"), cal_cfg.get("max_images_per_sku"))
    val_labels, _ = encode_labels(val["sku_id"])
    val_emb = _embed_frame(model, val, root, config["embedding"], device)
    calibrations = {}
    for method in methods:
        if method not in GRAPH_METHODS:
            continue
        calibration, diag = calibrate(method, model, val_emb, val_labels, cluster_cfg, seed)
        calibrations[method] = {**calibration, "pair_auc": diag["pair_auc"], "val_metrics": diag["best"],
                                "candidate_recall": diag["candidate_recall_of_positives"]}
        fig = pair_diagnostics_figure(diag["edge_features"], diag["edge_same"], calibration["features"],
                                      diag["threshold_sweep"], title=f"{model.name} / {method} / validation")
        fig.savefig(out_dir / f"validation_pairs_{method}.png", dpi=110)
        plt.close(fig)
    (out_dir / "calibration.json").write_text(json.dumps(calibrations, indent=2), encoding="utf-8")

    corpora = {c["name"]: select_records(manifest, c["split"], c.get("max_skus"), c.get("max_images_per_sku")) for c in config["corpora"]}
    union = pd.concat(corpora.values()).drop_duplicates("image_path").reset_index(drop=True)
    position = pd.Series(np.arange(len(union)), index=union["image_path"])
    test_emb = _embed_frame(model, union, root, config["embedding"], device)
    if config.get("save_embeddings"):
        np.savez_compressed(out_dir / "test_embeddings.npz", image_path=union["image_path"].to_numpy(),
                            sku_id=union["sku_id"].to_numpy(), **test_emb)

    rows = []
    limits = cluster_cfg.get("max_images_per_method") or {}
    for corpus_name, frame in corpora.items():
        rows_idx = position.loc[frame["image_path"]].to_numpy()
        emb = _subset(test_emb, rows_idx)
        labels, _ = encode_labels(frame["sku_id"])
        for method in methods:
            if limits.get(method) and len(frame) > int(limits[method]):
                print(f"  skip {method} on {corpus_name}: {len(frame)} images > limit {limits[method]}", flush=True)
                continue
            start = time.perf_counter()
            pred, extra = predict(method, model, emb, cluster_cfg, seed, calibrations.get(method), int(labels.max() + 1))
            seconds = time.perf_counter() - start
            row = {"model": model.name, "corpus": corpus_name, "method": method, "seconds": round(seconds, 2),
                   **clustering_metrics(labels, pred)}
            if "neighbors" in extra:
                row.update(recall_at_k(extra["neighbors"], labels))
            rows.append(row)
            print(f"  {corpus_name:<10} {method:<14} bcubed_f1={row['bcubed_f1']:.4f} "
                  f"clusters={row['n_pred_clusters']}/{row['n_true_clusters']} ({seconds:.1f}s)", flush=True)
            if method == methods[0]:
                fig = cluster_size_figure(pred, labels, title=f"{model.name} / {method} / {corpus_name}")
                fig.savefig(out_dir / f"cluster_sizes_{corpus_name}_{method}.png", dpi=110)
                plt.close(fig)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "metrics.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    save_yaml({k: v for k, v in config.items() if not k.startswith("_")}, out_dir / "evaluate_config.yaml")
    _log_mlflow(config, model, table, out_dir)
    return rows


def _log_mlflow(config: dict, model, table: pd.DataFrame, out_dir: Path) -> None:
    mlflow_cfg = config.get("mlflow") or {}
    if not mlflow_cfg.get("enabled", True):
        return
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment"])
    with mlflow.start_run(run_name=f"evaluate_{model.name}", tags={"stage": "evaluate", "model": model.name}):
        mlflow.log_param("checkpoint", str(model.checkpoint))
        for row in table.to_dict("records"):
            for key in ("bcubed_f1", "bcubed_precision", "bcubed_recall", "pairwise_f1", "nmi", "ari",
                        "cluster_accuracy", "cluster_count_ratio", "recall_at_1"):
                value = row.get(key)
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    mlflow.log_metric(f"{row['corpus']}/{row['method']}/{key}", float(value))
        mlflow.log_artifacts(str(out_dir), artifact_path=f"evaluation_{model.name}")


def evaluate(config: dict) -> pd.DataFrame:
    device = pick_device(config.get("device", "auto"))
    manifest = load_manifest(config["data"]["manifest"])
    rows = []
    for entry in config["models"]:
        rows.extend(evaluate_model(entry, config, manifest, device))
    summary = pd.DataFrame(rows)
    summary.to_csv(resolve_path(config["output_dir"]) / "evaluation_summary.csv", index=False)
    return summary
