"""Write clusters as folders: ``results/cluster_<model>/1, 2, 3, ...`` (largest cluster first)."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

MARKER = "summary.json"


def _prepare_output(directory: Path, overwrite: bool) -> None:
    if directory.exists() and any(directory.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{directory} is not empty. Set export.overwrite: true to replace it.")
        if not (directory / MARKER).is_file():
            raise FileExistsError(f"{directory} was not written by scripts/cluster.py ({MARKER} missing); refusing to delete it.")
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def folder_names(labels: np.ndarray, singletons: str) -> np.ndarray:
    """Map cluster labels to folder names. Clusters are numbered 1..N by size; ties by first image."""
    labels = np.asarray(labels)
    names = np.empty(len(labels), dtype=object)
    ids, first, counts = np.unique(labels, return_index=True, return_counts=True)
    multi = [(int(-c), int(f), int(i)) for i, f, c in zip(ids, first, counts) if c > 1 and i >= 0]
    rank = 0
    for _neg_count, _first, cluster in sorted(multi):
        rank += 1
        names[labels == cluster] = str(rank)
    lonely = np.flatnonzero(names == None)  # noqa: E711
    if singletons == "unassigned":
        names[lonely] = "unassigned"
    elif singletons == "own_folder":
        for offset, index in enumerate(lonely, start=rank + 1):
            names[index] = str(offset)
    else:
        raise ValueError("export.singletons must be unassigned or own_folder")
    return names


def link_strength(n: int, labels: np.ndarray, extra: dict) -> np.ndarray:
    """Strongest kNN edge probability that ties each image to its own cluster (NaN if none)."""
    strength = np.full(n, -np.inf)
    edges, probs = extra.get("edges"), extra.get("edge_probs")
    if edges is not None and len(edges):
        inside = labels[edges[:, 0]] == labels[edges[:, 1]]
        index = np.concatenate([edges[inside, 0], edges[inside, 1]])
        np.maximum.at(strength, index, np.concatenate([probs[inside], probs[inside]]))
    strength[np.isinf(strength)] = np.nan
    return strength


def export_clusters(paths: list[Path], labels: np.ndarray, extra: dict, cfg: dict, output_dir: Path, meta: dict) -> dict:
    mode = cfg.get("mode", "copy")
    if mode not in ("copy", "hardlink"):
        raise ValueError("export.mode must be copy or hardlink")
    _prepare_output(output_dir, bool(cfg.get("overwrite", False)))
    # Marker first, so an interrupted export can still be replaced by the next run.
    (output_dir / MARKER).write_text(json.dumps({"status": "incomplete", "created_by": "scripts/cluster.py"}), encoding="utf-8")
    names = folder_names(labels, cfg.get("singletons", "unassigned"))
    rows = []
    for index, (source, folder) in enumerate(zip(paths, names)):
        target_dir = output_dir / folder
        target_dir.mkdir(exist_ok=True)
        target = target_dir / f"{index:06d}_{source.name}"
        if mode == "hardlink":
            try:
                os.link(source, target)
            except OSError:  # different drive or a filesystem without hardlinks
                shutil.copy2(source, target)
        else:
            shutil.copy2(source, target)
        rows.append({"image": str(source), "cluster": folder, "exported_as": str(target.relative_to(output_dir))})
    table = pd.DataFrame(rows)
    sizes = table.groupby("cluster")["image"].transform("size")
    table["cluster_size"] = np.where(table["cluster"] == "unassigned", 1, sizes)
    table["link_probability"] = link_strength(len(paths), np.asarray(labels), extra)
    table.to_csv(output_dir / "assignments.csv", index=False, encoding="utf-8-sig")
    counts = pd.Series(names).value_counts()
    clustered = counts.drop("unassigned", errors="ignore")
    summary = {
        **meta,
        "images": int(len(paths)),
        "clusters": int(len(clustered)),
        "unassigned_images": int(counts.get("unassigned", 0)),
        "largest_cluster": int(clustered.max()) if len(clustered) else 0,
        "export_mode": mode,
        "created_by": "scripts/cluster.py",
    }
    (output_dir / MARKER).write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return summary
