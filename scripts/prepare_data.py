"""Build the SKU-disjoint manifest from configs/data.yaml."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    import json

    from src.config import load_config
    from src.data.splits import prepare_splits

    summary = prepare_splits(load_config("configs/data.yaml", sys.argv[1:]))
    print(json.dumps({k: summary[k] for k in ("manifest", "sha256", "skus", "images", "removed")}, indent=2))
