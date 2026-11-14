"""Cluster images into results/cluster_<model>/1..N folders with configs/cluster.yaml."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    import json

    from src.config import load_config
    from src.inference.run import run_clustering

    print(json.dumps(run_clustering(load_config("configs/cluster.yaml", sys.argv[1:])), indent=2, ensure_ascii=False, default=str))
