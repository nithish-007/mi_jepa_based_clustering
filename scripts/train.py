"""Train with configs/train.yaml. Re-running resumes from checkpoints/<name>/last.ckpt."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Imports live under __main__: on Windows every DataLoader worker re-runs this file's
# top level, and should not load Lightning, MLflow and timm just to read images.
if __name__ == "__main__":
    import json

    from src.config import load_config
    from src.training.trainer import train

    print(json.dumps(train(load_config("configs/train.yaml", sys.argv[1:])), indent=2))
