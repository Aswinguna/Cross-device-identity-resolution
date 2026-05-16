"""
MLflow Tracking Utilities
==========================
Thin wrappers that keep MLflow boilerplate out of the pipeline modules.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Optional

import mlflow

logger = logging.getLogger(__name__)


def setup_mlflow(cfg: dict) -> None:
    """Configure the MLflow tracking URI from config."""
    mlflow_cfg = cfg.get("mlflow", {})
    uri = mlflow_cfg.get("tracking_uri", "outputs/mlflow")
    mlflow.set_tracking_uri(uri)
    logger.info("MLflow tracking URI: %s", uri)


@contextmanager
def start_run(cfg: dict, experiment_key: str, run_name: Optional[str] = None):
    """
    Context manager that sets the experiment and starts a run.

    Parameters
    ----------
    cfg : full config dict
    experiment_key : "experiment_identity" | "experiment_segmentation"
    run_name : optional label shown in the MLflow UI

    Usage
    -----
    with start_run(cfg, "experiment_identity", run_name="rf_v1") as run:
        mlflow.log_param("model", "random_forest")
    """
    mlflow_cfg = cfg.get("mlflow", {})
    exp_name = mlflow_cfg.get(experiment_key, experiment_key)
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name=run_name) as run:
        logger.info(
            "MLflow run started | experiment=%s | run_id=%s",
            exp_name, run.info.run_id,
        )
        yield run
        logger.info("MLflow run finished | run_id=%s", run.info.run_id)


def log_dataset_info(df, tag: str = "sessions") -> None:
    """Log basic dataset statistics as MLflow params."""
    mlflow.log_params(
        {
            f"{tag}_n_rows": len(df),
            f"{tag}_n_cols": df.shape[1],
        }
    )
    if "user_id_hash" in df.columns:
        mlflow.log_param(f"{tag}_n_unique_users", df["user_id_hash"].nunique())
    if "device_type" in df.columns:
        for dev, cnt in df["device_type"].value_counts().items():
            mlflow.log_metric(f"{tag}_sessions_{dev}", int(cnt))


def log_figure(fig, filename: str) -> None:
    """Save a matplotlib figure as an MLflow artifact."""
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(path)
