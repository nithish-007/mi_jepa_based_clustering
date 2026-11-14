"""Evaluate clustering on test products with configs/evaluate.yaml."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    from src.config import load_config
    from src.evaluation.pipeline import evaluate

    summary = evaluate(load_config("configs/evaluate.yaml", sys.argv[1:]))
    columns = ["model", "corpus", "method", "bcubed_f1", "pairwise_f1", "nmi", "cluster_accuracy", "n_pred_clusters", "n_true_clusters"]
    print(summary[columns].to_string(index=False, float_format="%.4f"))
