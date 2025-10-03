"""Experiment tracking utilities (MLflow optional)."""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional


try:
    import mlflow  # type: ignore
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore


@contextmanager
def start_run(experiment: str, run_name: Optional[str] = None) -> Iterator[None]:
    if mlflow is None:
        yield
        return
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        yield


def log_params(params: Dict):
    if mlflow is None:
        return
    mlflow.log_params(params)


def log_metrics(metrics: Dict, step: Optional[int] = None):
    if mlflow is None:
        return
    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: Path, artifact_path: Optional[str] = None):
    if mlflow is None:
        return
    mlflow.log_artifact(str(path), artifact_path=artifact_path)


