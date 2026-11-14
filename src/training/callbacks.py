from __future__ import annotations

import json

import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
from mlflow.entities import Param

from src.evaluation.plots import pair_diagnostics_figure


def _flatten(config: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in config.items():
        if str(key).startswith("_"):
            continue
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = json.dumps(value) if isinstance(value, (list, tuple)) else str(value)
    return flat


class MLflowRunSetup(Callback):
    """Logs the YAML, flattened hyperparameters and the parameter table once per MLflow run."""

    def __init__(self, config: dict, resumed_from: str | None):
        self.config = config
        self.resumed_from = resumed_from

    def on_fit_start(self, trainer, pl_module):
        logger = trainer.logger
        if logger is None or not hasattr(logger, "run_id"):
            return
        client, run_id = logger.experiment, logger.run_id
        if self.resumed_from:
            client.set_tag(run_id, "resumed_from", self.resumed_from)
            return
        params = [Param(k, v[:6000]) for k, v in _flatten(self.config).items()]
        for start in range(0, len(params), 100):
            client.log_batch(run_id, params=params[start : start + 100])
        client.log_artifact(run_id, self.config["_source"], artifact_path="config")
        table = "\n".join(f"{r['module']:<28} {r['parameters']:>12,} {r['trainable']:>12,}" for r in pl_module.net.parameter_table())
        client.log_text(run_id, f"{'module':<28} {'parameters':>12} {'trainable':>12}\n{table}\n", "model/parameters.txt")
        client.log_text(run_id, str(pl_module.net), "model/architecture.txt")


class ValidationPlots(Callback):
    """Every N epochs, logs pair-score histograms and the threshold sweep to MLflow."""

    def __init__(self, every_n_epochs: int):
        self.every = int(every_n_epochs)

    def on_validation_end(self, trainer, pl_module):
        diag = pl_module.validation_diagnostics
        logger = trainer.logger
        if not self.every or diag is None or logger is None or trainer.sanity_checking:
            return
        epoch = diag["epoch"]
        if epoch % self.every:
            return
        names = pl_module.calibration.get("features", ["cosine", "neg_energy"])
        fig = pair_diagnostics_figure(diag["edge_features"], diag["edge_same"], names, diag["threshold_sweep"],
                                      title=f"validation, epoch {epoch}")
        logger.experiment.log_figure(logger.run_id, fig, f"plots/validation_epoch_{epoch:03d}.png")
        plt.close(fig)
